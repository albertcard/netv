/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * DNN TensorRT backend implementation.
 *
 * This backend loads pre-compiled TensorRT engine files (.engine) for
 * high-performance GPU inference. Use tools/export-tensorrt.py to convert
 * PyTorch models to TensorRT engines.
 *
 * Usage:
 *   ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model.engine:input=input:output=output" output.mp4
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>

extern "C" {
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "queue.h"
#include "safe_queue.h"
}

// TensorRT logger - forward to FFmpeg's logging
class TRTLogger : public nvinfer1::ILogger {
public:
    void *log_ctx;
    TRTLogger(void *ctx = nullptr) : log_ctx(ctx) {}

    void log(Severity severity, const char *msg) noexcept override {
        int level;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                level = AV_LOG_ERROR;
                break;
            case Severity::kWARNING:
                level = AV_LOG_WARNING;
                break;
            case Severity::kINFO:
                level = AV_LOG_INFO;
                break;
            default:
                level = AV_LOG_DEBUG;
                break;
        }
        av_log(log_ctx, level, "TensorRT: %s\n", msg);
    }
};

typedef struct TRTModel {
    DNNModel model;
    DnnContext *ctx;
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    TRTLogger *logger;
    cudaStream_t stream;

    // I/O buffer info
    int input_index;
    int output_index;
    nvinfer1::Dims input_dims;
    nvinfer1::Dims output_dims;

    // CUDA buffers
    void *input_buffer;
    void *output_buffer;
    size_t input_size;
    size_t output_size;

    // Task management (reuse FFmpeg's queue infrastructure)
    SafeQueue *request_queue;
    Queue *task_queue;
    Queue *lltask_queue;
} TRTModel;

typedef struct TRTInferRequest {
    float *output_data;  // CPU output buffer
} TRTInferRequest;

typedef struct TRTRequestItem {
    TRTInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    DNNAsyncExecModule exec_module;
} TRTRequestItem;

#define OFFSET(x) offsetof(TRTOptions, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM

typedef struct TRTOptions {
    const AVClass *clazz;
    int device_id;  // CUDA device ID
} TRTOptions;

static const AVOption dnn_trt_options[] = {
    { "device_id", "CUDA device ID", OFFSET(device_id), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 16, FLAGS },
    { NULL }
};

// Check CUDA error and log
#define CUDA_CHECK(call, ctx, ret) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        av_log(ctx, AV_LOG_ERROR, "CUDA error: %s\n", cudaGetErrorString(err)); \
        return ret; \
    } \
} while(0)

static int extract_lltask_from_task(TaskItem *task, Queue *lltask_queue)
{
    TRTModel *trt_model = (TRTModel *)task->model;
    DnnContext *ctx = trt_model->ctx;
    LastLevelTaskItem *lltask = (LastLevelTaskItem *)av_malloc(sizeof(*lltask));
    if (!lltask) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for LastLevelTaskItem\n");
        return AVERROR(ENOMEM);
    }
    task->inference_todo = 1;
    task->inference_done = 0;
    lltask->task = task;
    if (ff_queue_push_back(lltask_queue, lltask) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
        av_freep(&lltask);
        return AVERROR(ENOMEM);
    }
    return 0;
}

static void trt_free_request(TRTInferRequest *request)
{
    if (!request)
        return;
    if (request->output_data) {
        av_freep(&request->output_data);
    }
}

static inline void destroy_request_item(TRTRequestItem **arg)
{
    TRTRequestItem *item;
    if (!arg || !*arg)
        return;
    item = *arg;
    trt_free_request(item->infer_request);
    av_freep(&item->infer_request);
    av_freep(&item->lltask);
    ff_dnn_async_module_cleanup(&item->exec_module);
    av_freep(arg);
}

static void dnn_free_model_trt(DNNModel **model)
{
    TRTModel *trt_model;
    if (!model || !*model)
        return;

    trt_model = (TRTModel *)(*model);

    // Free CUDA resources
    if (trt_model->input_buffer) {
        cudaFree(trt_model->input_buffer);
        trt_model->input_buffer = nullptr;
    }
    if (trt_model->output_buffer) {
        cudaFree(trt_model->output_buffer);
        trt_model->output_buffer = nullptr;
    }
    if (trt_model->stream) {
        cudaStreamDestroy(trt_model->stream);
        trt_model->stream = nullptr;
    }

    // Free TensorRT resources
    if (trt_model->context) {
        delete trt_model->context;
        trt_model->context = nullptr;
    }
    if (trt_model->engine) {
        delete trt_model->engine;
        trt_model->engine = nullptr;
    }
    if (trt_model->runtime) {
        delete trt_model->runtime;
        trt_model->runtime = nullptr;
    }
    if (trt_model->logger) {
        delete trt_model->logger;
        trt_model->logger = nullptr;
    }

    // Free queues
    if (trt_model->request_queue) {
        while (ff_safe_queue_size(trt_model->request_queue) != 0) {
            TRTRequestItem *item = (TRTRequestItem *)ff_safe_queue_pop_front(trt_model->request_queue);
            destroy_request_item(&item);
        }
        ff_safe_queue_destroy(trt_model->request_queue);
    }
    if (trt_model->lltask_queue) {
        while (ff_queue_size(trt_model->lltask_queue) != 0) {
            LastLevelTaskItem *item = (LastLevelTaskItem *)ff_queue_pop_front(trt_model->lltask_queue);
            av_freep(&item);
        }
        ff_queue_destroy(trt_model->lltask_queue);
    }
    if (trt_model->task_queue) {
        while (ff_queue_size(trt_model->task_queue) != 0) {
            TaskItem *item = (TaskItem *)ff_queue_pop_front(trt_model->task_queue);
            av_frame_free(&item->in_frame);
            av_frame_free(&item->out_frame);
            av_freep(&item);
        }
        ff_queue_destroy(trt_model->task_queue);
    }

    av_freep(&trt_model);
    *model = NULL;
}

static int get_input_trt(DNNModel *model, DNNData *input, const char *input_name)
{
    TRTModel *trt_model = (TRTModel *)model;

    input->dt = DNN_FLOAT;
    input->order = DCO_RGB;
    input->layout = DL_NCHW;

    // Get dimensions from engine
    input->dims[0] = trt_model->input_dims.d[0];  // N (batch)
    input->dims[1] = trt_model->input_dims.d[1];  // C (channels)
    input->dims[2] = trt_model->input_dims.d[2];  // H (height)
    input->dims[3] = trt_model->input_dims.d[3];  // W (width)

    return 0;
}

static void deleter(void *arg)
{
    av_freep(&arg);
}

static int fill_model_input_trt(TRTModel *trt_model, TRTRequestItem *request)
{
    LastLevelTaskItem *lltask = NULL;
    TaskItem *task = NULL;
    TRTInferRequest *infer_request = NULL;
    DNNData input = { 0 };
    DnnContext *ctx = trt_model->ctx;
    int ret;

    lltask = (LastLevelTaskItem *)ff_queue_pop_front(trt_model->lltask_queue);
    if (!lltask) {
        return AVERROR(EINVAL);
    }
    request->lltask = lltask;
    task = lltask->task;
    infer_request = request->infer_request;

    ret = get_input_trt(&trt_model->model, &input, NULL);
    if (ret != 0) {
        return ret;
    }

    int height_idx = dnn_get_height_idx_by_layout(input.layout);
    int width_idx = dnn_get_width_idx_by_layout(input.layout);
    int channel_idx = dnn_get_channel_idx_by_layout(input.layout);

    // Check input dimensions match engine
    if (task->in_frame->height != input.dims[height_idx] ||
        task->in_frame->width != input.dims[width_idx]) {
        av_log(ctx, AV_LOG_ERROR, "Input size %dx%d doesn't match engine's expected %dx%d\n",
               task->in_frame->width, task->in_frame->height,
               input.dims[width_idx], input.dims[height_idx]);
        return AVERROR(EINVAL);
    }

    // Allocate CPU buffer for input preprocessing
    size_t input_elements = input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3];
    float *input_data = (float *)av_malloc(input_elements * sizeof(float));
    if (!input_data) {
        return AVERROR(ENOMEM);
    }

    input.data = input_data;
    input.scale = 255;

    switch (trt_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            if (trt_model->model.frame_pre_proc != NULL) {
                trt_model->model.frame_pre_proc(task->in_frame, &input, trt_model->model.filter_ctx);
            } else {
                ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
            }
        }
        break;
    default:
        avpriv_report_missing_feature(NULL, "model function type %d", trt_model->model.func_type);
        av_freep(&input_data);
        return AVERROR(EINVAL);
    }

    // Copy input to GPU
    CUDA_CHECK(cudaMemcpyAsync(trt_model->input_buffer, input_data, trt_model->input_size,
                               cudaMemcpyHostToDevice, trt_model->stream), ctx, AVERROR(EIO));

    av_freep(&input_data);
    return 0;
}

static int trt_start_inference(void *args)
{
    TRTRequestItem *request = (TRTRequestItem *)args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    TRTModel *trt_model = (TRTModel *)task->model;
    DnnContext *ctx = trt_model->ctx;

    // Set up I/O bindings
    void *bindings[2];
    bindings[trt_model->input_index] = trt_model->input_buffer;
    bindings[trt_model->output_index] = trt_model->output_buffer;

    // Run inference
    bool success = trt_model->context->enqueueV2(bindings, trt_model->stream, nullptr);
    if (!success) {
        av_log(ctx, AV_LOG_ERROR, "TensorRT inference failed\n");
        return DNN_GENERIC_ERROR;
    }

    // Synchronize
    CUDA_CHECK(cudaStreamSynchronize(trt_model->stream), ctx, DNN_GENERIC_ERROR);

    return 0;
}

static void infer_completion_callback(void *args)
{
    TRTRequestItem *request = (TRTRequestItem *)args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    TRTModel *trt_model = (TRTModel *)task->model;
    DnnContext *ctx = trt_model->ctx;
    DNNData outputs = { 0 };

    outputs.order = DCO_RGB;
    outputs.layout = DL_NCHW;
    outputs.dt = DNN_FLOAT;
    outputs.dims[0] = trt_model->output_dims.d[0];  // N
    outputs.dims[1] = trt_model->output_dims.d[1];  // C
    outputs.dims[2] = trt_model->output_dims.d[2];  // H
    outputs.dims[3] = trt_model->output_dims.d[3];  // W

    // Allocate CPU buffer for output
    size_t output_elements = outputs.dims[0] * outputs.dims[1] * outputs.dims[2] * outputs.dims[3];
    float *output_data = (float *)av_malloc(output_elements * sizeof(float));
    if (!output_data) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate output buffer\n");
        goto err;
    }

    // Copy output from GPU
    cudaError_t err = cudaMemcpy(output_data, trt_model->output_buffer, trt_model->output_size,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        av_log(ctx, AV_LOG_ERROR, "Failed to copy output from GPU: %s\n", cudaGetErrorString(err));
        av_freep(&output_data);
        goto err;
    }

    switch (trt_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            outputs.scale = 255;
            outputs.data = output_data;
            if (trt_model->model.frame_post_proc != NULL) {
                trt_model->model.frame_post_proc(task->out_frame, &outputs, trt_model->model.filter_ctx);
            } else {
                ff_proc_from_dnn_to_frame(task->out_frame, &outputs, ctx);
            }
        } else {
            task->out_frame->width = outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];
            task->out_frame->height = outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
        }
        break;
    default:
        avpriv_report_missing_feature(ctx, "model function type %d", trt_model->model.func_type);
        av_freep(&output_data);
        goto err;
    }

    av_freep(&output_data);
    task->inference_done++;

err:
    av_freep(&request->lltask);
    if (ff_safe_queue_push_back(trt_model->request_queue, request) < 0) {
        destroy_request_item(&request);
        av_log(ctx, AV_LOG_ERROR, "Unable to push back request_queue.\n");
    }
}

static int execute_model_trt(TRTRequestItem *request, Queue *lltask_queue)
{
    TRTModel *trt_model = NULL;
    LastLevelTaskItem *lltask;
    TaskItem *task = NULL;
    int ret = 0;

    if (ff_queue_size(lltask_queue) == 0) {
        destroy_request_item(&request);
        return 0;
    }

    lltask = (LastLevelTaskItem *)ff_queue_peek_front(lltask_queue);
    if (lltask == NULL) {
        av_log(NULL, AV_LOG_ERROR, "Failed to get LastLevelTaskItem\n");
        ret = AVERROR(EINVAL);
        goto err;
    }
    task = lltask->task;
    trt_model = (TRTModel *)task->model;

    ret = fill_model_input_trt(trt_model, request);
    if (ret != 0) {
        goto err;
    }

    // Synchronous execution (TensorRT is fast, async adds complexity)
    ret = trt_start_inference((void *)request);
    if (ret != 0) {
        goto err;
    }
    infer_completion_callback(request);
    return (task->inference_done == task->inference_todo) ? 0 : DNN_GENERIC_ERROR;

err:
    trt_free_request(request->infer_request);
    if (ff_safe_queue_push_back(trt_model->request_queue, request) < 0) {
        destroy_request_item(&request);
    }
    return ret;
}

static int get_output_trt(DNNModel *model, const char *input_name, int input_width, int input_height,
                          const char *output_name, int *output_width, int *output_height)
{
    TRTModel *trt_model = (TRTModel *)model;

    // For super-resolution, output is typically 4x input
    // Get from engine's output dimensions
    *output_width = trt_model->output_dims.d[3];
    *output_height = trt_model->output_dims.d[2];

    return 0;
}

static TRTInferRequest *trt_create_inference_request(void)
{
    TRTInferRequest *request = (TRTInferRequest *)av_mallocz(sizeof(TRTInferRequest));
    return request;
}

static DNNModel *dnn_load_model_trt(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    TRTModel *trt_model = NULL;
    TRTRequestItem *item = NULL;

    trt_model = (TRTModel *)av_mallocz(sizeof(TRTModel));
    if (!trt_model)
        return NULL;

    trt_model->ctx = ctx;

    // Create TensorRT logger
    trt_model->logger = new TRTLogger(ctx);

    // Create runtime
    trt_model->runtime = nvinfer1::createInferRuntime(*trt_model->logger);
    if (!trt_model->runtime) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create TensorRT runtime\n");
        goto fail;
    }

    // Load engine from file
    {
        std::ifstream file(ctx->model_filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            av_log(ctx, AV_LOG_ERROR, "Failed to open engine file: %s\n", ctx->model_filename);
            goto fail;
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            av_log(ctx, AV_LOG_ERROR, "Failed to read engine file\n");
            goto fail;
        }

        trt_model->engine = trt_model->runtime->deserializeCudaEngine(buffer.data(), size);
        if (!trt_model->engine) {
            av_log(ctx, AV_LOG_ERROR, "Failed to deserialize CUDA engine\n");
            goto fail;
        }
    }

    // Create execution context
    trt_model->context = trt_model->engine->createExecutionContext();
    if (!trt_model->context) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create execution context\n");
        goto fail;
    }

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&trt_model->stream), ctx, NULL);

    // Get I/O binding info
    {
        int nb_bindings = trt_model->engine->getNbBindings();
        if (nb_bindings < 2) {
            av_log(ctx, AV_LOG_ERROR, "Engine must have at least 2 bindings (input and output)\n");
            goto fail;
        }

        // Find input and output indices
        trt_model->input_index = -1;
        trt_model->output_index = -1;
        for (int i = 0; i < nb_bindings; i++) {
            if (trt_model->engine->bindingIsInput(i)) {
                if (trt_model->input_index < 0) {
                    trt_model->input_index = i;
                    trt_model->input_dims = trt_model->engine->getBindingDimensions(i);
                }
            } else {
                if (trt_model->output_index < 0) {
                    trt_model->output_index = i;
                    trt_model->output_dims = trt_model->engine->getBindingDimensions(i);
                }
            }
        }

        if (trt_model->input_index < 0 || trt_model->output_index < 0) {
            av_log(ctx, AV_LOG_ERROR, "Could not find input/output bindings\n");
            goto fail;
        }

        // Log dimensions
        av_log(ctx, AV_LOG_INFO, "TensorRT engine loaded: input %dx%dx%dx%d, output %dx%dx%dx%d\n",
               trt_model->input_dims.d[0], trt_model->input_dims.d[1],
               trt_model->input_dims.d[2], trt_model->input_dims.d[3],
               trt_model->output_dims.d[0], trt_model->output_dims.d[1],
               trt_model->output_dims.d[2], trt_model->output_dims.d[3]);

        // Allocate GPU buffers
        trt_model->input_size = trt_model->input_dims.d[0] * trt_model->input_dims.d[1] *
                                 trt_model->input_dims.d[2] * trt_model->input_dims.d[3] * sizeof(float);
        trt_model->output_size = trt_model->output_dims.d[0] * trt_model->output_dims.d[1] *
                                  trt_model->output_dims.d[2] * trt_model->output_dims.d[3] * sizeof(float);

        CUDA_CHECK(cudaMalloc(&trt_model->input_buffer, trt_model->input_size), ctx, NULL);
        CUDA_CHECK(cudaMalloc(&trt_model->output_buffer, trt_model->output_size), ctx, NULL);
    }

    // Initialize queues
    trt_model->request_queue = ff_safe_queue_create();
    if (!trt_model->request_queue)
        goto fail;

    item = (TRTRequestItem *)av_mallocz(sizeof(TRTRequestItem));
    if (!item)
        goto fail;

    item->infer_request = trt_create_inference_request();
    if (!item->infer_request)
        goto fail;

    item->exec_module.start_inference = &trt_start_inference;
    item->exec_module.callback = &infer_completion_callback;
    item->exec_module.args = item;

    if (ff_safe_queue_push_back(trt_model->request_queue, item) < 0)
        goto fail;
    item = NULL;

    trt_model->task_queue = ff_queue_create();
    if (!trt_model->task_queue)
        goto fail;

    trt_model->lltask_queue = ff_queue_create();
    if (!trt_model->lltask_queue)
        goto fail;

    // Set up model interface
    trt_model->model.get_input = &get_input_trt;
    trt_model->model.get_output = &get_output_trt;
    trt_model->model.filter_ctx = filter_ctx;
    trt_model->model.func_type = func_type;

    return &trt_model->model;

fail:
    if (item) {
        destroy_request_item(&item);
    }
    dnn_free_model_trt((DNNModel **)&trt_model);
    return NULL;
}

static int dnn_execute_model_trt(const DNNModel *model, DNNExecBaseParams *exec_params)
{
    TRTModel *trt_model = (TRTModel *)model;
    DnnContext *ctx = trt_model->ctx;
    TaskItem *task;
    TRTRequestItem *request;
    int ret = 0;

    ret = ff_check_exec_params(ctx, DNN_TRT, model->func_type, exec_params);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "exec parameter checking fail.\n");
        return ret;
    }

    task = (TaskItem *)av_malloc(sizeof(TaskItem));
    if (!task) {
        av_log(ctx, AV_LOG_ERROR, "unable to alloc memory for task item.\n");
        return AVERROR(ENOMEM);
    }

    ret = ff_dnn_fill_task(task, exec_params, trt_model, 0, 1);
    if (ret != 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to fill task.\n");
        return ret;
    }

    ret = ff_queue_push_back(trt_model->task_queue, task);
    if (ret < 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        return ret;
    }

    ret = extract_lltask_from_task(task, trt_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        return ret;
    }

    request = (TRTRequestItem *)ff_safe_queue_pop_front(trt_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    return execute_model_trt(request, trt_model->lltask_queue);
}

static DNNAsyncStatusType dnn_get_result_trt(const DNNModel *model, AVFrame **in, AVFrame **out)
{
    TRTModel *trt_model = (TRTModel *)model;
    return ff_dnn_get_result_common(trt_model->task_queue, in, out);
}

static int dnn_flush_trt(const DNNModel *model)
{
    TRTModel *trt_model = (TRTModel *)model;
    TRTRequestItem *request;

    if (ff_queue_size(trt_model->lltask_queue) == 0)
        return 0;

    request = (TRTRequestItem *)ff_safe_queue_pop_front(trt_model->request_queue);
    if (!request) {
        av_log(trt_model->ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    return execute_model_trt(request, trt_model->lltask_queue);
}

extern const DNNModule ff_dnn_backend_tensorrt = {
    .clazz          = DNN_DEFINE_CLASS(dnn_trt),
    .type           = DNN_TRT,
    .load_model     = dnn_load_model_trt,
    .execute_model  = dnn_execute_model_trt,
    .get_result     = dnn_get_result_trt,
    .flush          = dnn_flush_trt,
    .free_model     = dnn_free_model_trt,
};
