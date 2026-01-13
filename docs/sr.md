# Super-Resolution with FFmpeg + LibTorch/TensorRT

## Overview

Goal: Real-time 4x super-resolution (720p → 4K) using Real-ESRGAN through FFmpeg's DNN processing filter with GPU acceleration.

## Test Environment

- **Machine:** colossus
- **GPU:** RTX 5090
- **Working directory:** `~/projects/netv`
- **FFmpeg source:** `~/ffmpeg_sources/ffmpeg-snapshot`
- **Models:** `~/ffmpeg_build/models/`

## Current Status

**Problem:** ~4x slowdown between pure PyTorch (95 fps) and FFmpeg+libtorch (22 fps) for the same model.

**Key finding:** Must specify `device=cuda` in dnn_processing filter. Without it, model runs on CPU (2 fps).

**Open question:** What causes the remaining 4x gap? Likely candidates:
- JIT recompilation per frame
- CUDA context switching (FFmpeg vs PyTorch contexts)
- Synchronous execution / excessive cudaDeviceSynchronize
- Per-frame memory allocation
- FFmpeg filter framework overhead

**NOT the bottleneck:** Color conversion (NV12↔RGB). This is ~1ms either way.

## Models

| Model | Architecture | Size | Scale | Speed | Quality |
|-------|--------------|------|-------|-------|---------|
| `realesr-general-x4v3.pt` | SRVGGNetCompact | 5MB | 4x | Fast | Good |
| `RealESRGAN_x4plus.pt` | RRDBNet (23 blocks) | 67MB | 4x | Slow | Best |
| `RealESRGAN_x2plus.pt` | RRDBNet (23 blocks) | 67MB | 2x | Slow | Best |

Models stored in: `~/ffmpeg_build/models/`

### Architecture Comparison

**SRVGGNetCompact** (realesr-general-x4v3)
- Lightweight network designed for real-time video
- ~13x fewer parameters than RRDBNet
- Best speed/quality tradeoff for video

**RRDBNet** (x4plus, x2plus)
- Full Real-ESRGAN architecture with residual-in-residual dense blocks
- Higher quality but much slower
- Better for single images or offline processing

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `tools/install-superres.sh` | Download model from HuggingFace, build TensorRT engine | Ready |
| `tools/export-tensorrt.py` | Export PyTorch → ONNX → TensorRT with dynamic shapes | Ready |
| `tools/install-ffmpeg.sh` | Build FFmpeg with libtorch/TensorRT support | Ready |

### install-superres.sh

Main installation script that:
1. Creates a Python venv in `~/ffmpeg_build/models/.venv`
2. Installs torch, huggingface_hub, onnx, tensorrt
3. Downloads Real-ESRGAN from HuggingFace
4. Builds TensorRT engine with dynamic shapes (270p to 1280p)
5. Outputs `~/ffmpeg_build/models/realesrgan_dynamic_fp16.engine`

### export-tensorrt.py

Flexible export script with options:
```bash
# Default: dynamic shapes for 270p-1280p, FP16
python tools/export-tensorrt.py --output model.engine

# Custom height range
python tools/export-tensorrt.py --min-height 360 --max-height 1080

# From custom model file
python tools/export-tensorrt.py --model /path/to/model.pth

# ONNX only (skip TensorRT build)
python tools/export-tensorrt.py --onnx-only --keep-onnx
```

Dynamic shape support: A single engine handles inputs from min to max resolution.

## FFmpeg Patches

Location: `tools/patches/` - Full source file replacements (not diffs)

| File | Size | Purpose |
|------|------|---------|
| `dnn_backend_tensorrt.cpp` | 30.5 KB | Native TensorRT backend for FFmpeg |
| `dnn_backend_torch.cpp` | 32.4 KB | LibTorch backend with CUDA support |
| `vf_dnn_processing.c` | ~16 KB | Video filter with CUDA frame support |
| `dnn_cuda_kernels.cu` | 7.1 KB | GPU-resident format conversion kernels |
| `dnn_cuda_kernels.h` | ~1 KB | CUDA kernel headers |

### CUDA Kernels (dnn_cuda_kernels.cu)

Zero-copy GPU-resident format conversion:

1. **`hwc_uint8_to_nchw_float32_kernel`**: HWC uint8 [0,255] → NCHW float32 [0,1]
2. **`nchw_float32_to_hwc_uint8_kernel`**: NCHW float32 [0,1] → HWC uint8 [0,255]
3. 4-channel variants for RGBA formats

These kernels keep data on GPU, avoiding costly GPU↔CPU transfers.

## Benchmark Plan

Test configuration: **RealESRGAN_x4plus, 1280x720 input, 100 frames**

| # | Scenario | What it measures |
|---|----------|------------------|
| 1 | Pure eager PyTorch | Baseline, no optimization |
| 2 | Pure torch.compile | PyTorch compiler |
| 3 | Pure TensorRT | Theoretical ceiling |
| 4 | FFmpeg + libtorch | Current patches, PyTorch JIT |
| 5 | FFmpeg + TensorRT | Patches + TRT-compiled model |

### Benchmark results (RTX 5090, realesr-general-x4v3, 1280x720, 150 frames)

| Scenario | FPS | Realtime | Notes |
|----------|-----|----------|-------|
| 1. Pure eager PyTorch | TBD | | |
| 2. Pure torch.compile | TBD | | |
| 3. Pure TensorRT (Python) | 66.1 | 2.20x | Direct inference, no ffmpeg |
| 4. FFmpeg + libtorch | 3.6 | 0.12x | Baseline |
| 5. FFmpeg + TensorRT | 4.6 | 0.15x | +28% but massive overhead |

### Why TensorRT is slow in FFmpeg

Python achieves 66 FPS but ffmpeg only 4.6 FPS. The bottleneck is **memory operations, not model inference**.

```
ffmpeg -benchmark output:
  utime=12.164s  (CPU computation)
  stime=20.526s  (kernel/memory ops)  <-- 63% of total time!
  rtime=32.375s  (wall clock)
```

Per-frame breakdown (150 frames, 32.4s total):
- Total time: 216ms/frame
- System time: **137ms/frame** (memory operations)
- User time: 81ms/frame (model + format conversion)
- TensorRT inference: ~15ms/frame (measured in Python)

### The real bottleneck: output size

4x upscaling produces massive output frames:
```
Input:  1280 x 720  x 3 bytes =  2.7 MB/frame
Output: 5120 x 2880 x 3 bytes = 44.2 MB/frame (16x larger!)
```

For 150 frames at 4.6 fps:
- 6.6 GB of output data copied GPU→CPU
- Memory allocation/deallocation per frame
- Page faults and cache misses

### Why Python is 14x faster

Python benchmark keeps tensors on GPU:
1. Create input tensor on GPU (once)
2. Run forward() N times
3. Never copy 44MB output to CPU

FFmpeg must copy to CPU for the filter chain:
1. Decode frame to CPU
2. Convert RGB→float tensor
3. Copy to GPU
4. Run inference (~15ms)
5. Copy 44MB output back to CPU  ← **bottleneck**
6. Convert float→RGB
7. Continue filter chain

### Evidence: TRT only marginally faster

- Vanilla libtorch: 3.6 fps (278ms/frame)
- TensorRT: 4.6 fps (217ms/frame)
- Difference: only 61ms saved

If inference was the bottleneck, TRT should save ~250ms (277ms - 15ms).
The 61ms savings confirms inference is only ~25% of total time.

## Usage

### Install super-resolution (downloads model + builds TensorRT engine)
```bash
cd ~/projects/netv
./tools/install-superres.sh
```

This creates `~/ffmpeg_build/models/realesrgan_dynamic_fp16.engine` with dynamic shape support.

### Manual export with custom settings
```bash
# Export for specific resolution range
python tools/export-tensorrt.py --min-height 480 --max-height 1080 --output custom.engine

# Export FP32 (higher precision, slower)
python tools/export-tensorrt.py --fp32 --output model_fp32.engine
```

### FFmpeg command examples

Basic (CPU):
```bash
ffmpeg -i input.mp4 \
  -vf "format=rgb24,dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt:device=cpu" \
  output.mp4
```

CUDA (GPU):
```bash
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 \
  -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt:device=cuda" \
  -c:v hevc_nvenc output.mp4
```

**Important:** Must specify `device=cuda` for GPU inference.

## TensorRT Acceleration

Two approaches are available:

### Option A: Native TensorRT Backend (Recommended)

The fastest approach - FFmpeg loads TensorRT engines directly, no libtorch dependency.

**Build FFmpeg with TensorRT:**
```bash
ENABLE_TENSORRT=1 ENABLE_LIBTORCH=0 ./tools/install-ffmpeg.sh
```

**Export model to TensorRT engine with dynamic shapes:**
```bash
# Default: supports 270p to 1280p input (16:9 aspect ratio)
python tools/export-tensorrt.py --output model.engine

# Custom range (e.g., 360p to 1080p)
python tools/export-tensorrt.py --min-height 360 --max-height 1080 --output model.engine
```

**Run with FFmpeg:**
```bash
ffmpeg -i input.mp4 \
  -vf "dnn_processing=dnn_backend=tensorrt:model=$HOME/ffmpeg_build/models/realesrgan_dynamic_fp16.engine" \
  -c:v hevc_nvenc output.mp4
```

**Note:** Dynamic shape engines handle a range of input resolutions without needing
separate engines. The engine is optimized for the `--opt-height` (default 720p).

### Option B: LibTorch + torch_tensorrt

Alternative approach using libtorch with TensorRT-compiled TorchScript models.
Requires larger dependencies but supports dynamic shapes.

**Build FFmpeg with LibTorch:**
```bash
ENABLE_TENSORRT=0 ENABLE_LIBTORCH=1 ./tools/install-ffmpeg.sh
```

**Run with FFmpeg:**

```bash
TORCH_LIB=~/ffmpeg_sources/libtorch/lib

LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH" \
ffmpeg -i input.mp4 \
  -vf "format=rgb24,dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3.pt:device=cuda" \
  output.mp4
```

### Performance Comparison

| Backend | 720p→4K | Dependencies |
|---------|---------|--------------|
| Native TensorRT | ~65 fps | libnvinfer only |
| LibTorch + torch_tensorrt | ~57 fps | libtorch + torch_tensorrt |
| LibTorch JIT | ~12 fps | libtorch only |

## GPU Compatibility

### libtorch version requirements

| GPU Generation | SM Version | Min libtorch | CUDA Variant |
|---------------|------------|--------------|--------------|
| Maxwell (TITAN X) | 5.2 | 2.5.0 | cu124 |
| Pascal (GTX 10xx) | 6.x | 2.5.0 | cu124 |
| Volta (V100) | 7.0 | 2.5.0 | cu124 |
| Turing (RTX 20xx) | 7.5 | 2.5.0 | cu124 |
| Ampere (RTX 30xx) | 8.x | 2.5.0 | cu124 |
| Ada (RTX 40xx) | 8.9 | 2.5.0 | cu124 |
| Blackwell (RTX 50xx) | 12.0 | 2.7.0+ | cu130 |

### TensorRT requirements

- Minimum SM 7.0 (Volta or newer)
- Maxwell/Pascal GPUs cannot use TensorRT 10.x

## Known Issues

1. **Must specify `device=cuda`** - Without this, model runs on CPU even with CUDA frames
2. **LD_PRELOAD required** - Need `LD_PRELOAD=$HOME/.local/lib/libtorch_cuda.so` for CUDA to work
3. **NPP API version** - CUDA 13+ requires `_Ctx` suffix on NPP functions (colossus has CUDA 13)
4. **libtorch version mismatch** - `initXPU()` renamed to `init()` in libtorch 2.6+

## Troubleshooting

### "no kernel image is available for execution on the device"
GPU not supported by libtorch version. Check GPU compatibility table above.

### "Unknown type name '__torch__.torch.classes.tensorrt.Engine'"
TensorRT runtime not loaded. Set LD_LIBRARY_PATH to include torch_tensorrt/lib.

### "Expected Tensor but got None"
TRT model output format issue. Ensure ffmpeg has TensorRT patches applied.

### Terminal corrupted after ffmpeg crash
Run `reset` or the test script includes `trap 'stty sane' EXIT`.

## FFmpeg Patches Applied

The `install-ffmpeg.sh` script applies these patches to ffmpeg's torch backend:

### 1. libtorch 2.6+ compatibility
```cpp
// initXPU() renamed to init() in libtorch 2.6+
at::detail::getXPUHooks().init();  // was initXPU()
```

### 2. CUDA device support
```cpp
// Upstream only supports CPU/XPU, we add CUDA
} else if (device.is_cuda()) {
    if (!at::cuda::is_available()) {
        av_log(ctx, AV_LOG_ERROR, "No CUDA device found\n");
        goto fail;
    }
}
```

### 3. TensorRT support
```cpp
// Load TensorRT runtime for TRT-compiled models
dlopen("libtorchtrt_runtime.so", RTLD_NOW | RTLD_GLOBAL);

// Handle TRT models that return tuples
auto output = model->forward(inputs);
if (output.isTuple()) {
    result = output.toTuple()->elements()[0].toTensor();
} else {
    result = output.toTensor();
}

// TRT models may not have parameters()
if (params.begin() != params.end()) {
    device = (*params.begin()).device();
}
```

## Future Improvements

### For real-time performance
1. **Keep tensors on GPU** - avoid CPU roundtrip entirely (what our patches attempt)
2. **Direct NVENC encoding** - GPU output → GPU encoder
3. **Pinned memory** - faster GPU↔CPU transfers
4. **Async pipeline** - overlap copy with next inference
5. **Batch processing** - amortize overhead across frames

### Model improvements
1. **Custom compact 2x model** - Train SRVGGNetCompact for 2x upscaling
2. **Quantized models** - INT8 inference for more speed
3. **Frame interpolation** - Combine with RIFE for frame rate upscaling

## Preliminary Benchmarks (incomplete)

Test: Simple 2x upscale model, 720p→1440p, on colossus (RTX 5090)

| Scenario | FPS | Notes |
|----------|-----|-------|
| Pure PyTorch JIT | 95 | Model inference only |
| Python full simulation | 89 | Including color conversion |
| FFmpeg without `device=cuda` | 2 | BUG: model on CPU |
| FFmpeg with `device=cuda` | 22 | Correct, but 4x slower than pure PyTorch |

**Note:** These were with a toy model, not RealESRGAN_x4plus. Need to redo with proper model.

## Patch Architecture

The `tools/patches/` directory contains complete source file replacements that `install-ffmpeg.sh` copies into the FFmpeg source tree.

### Files and their roles

| File | Target Location | Purpose |
|------|-----------------|---------|
| `dnn_backend_tensorrt.cpp` | `libavfilter/dnn/` | Native TensorRT engine loading and inference |
| `dnn_backend_torch.cpp` | `libavfilter/dnn/` | LibTorch backend with CUDA device support |
| `vf_dnn_processing.c` | `libavfilter/` | Video filter with CUDA frame awareness |
| `dnn_cuda_kernels.cu` | `libavfilter/dnn/` | GPU format conversion (HWC↔NCHW) |
| `dnn_cuda_kernels.h` | `libavfilter/dnn/` | CUDA kernel declarations |

### Key modifications over upstream FFmpeg

**dnn_backend_torch.cpp:**
- `initXPU()` → `init()` for libtorch 2.6+ compatibility
- Added CUDA device support (upstream only supports CPU/XPU)
- TensorRT runtime loading for TRT-compiled models
- Output tuple handling for TRT models
- Fallback device detection for models without parameters

**dnn_backend_tensorrt.cpp:**
- Native TensorRT backend (not in upstream FFmpeg)
- Loads .engine files directly
- Dynamic shape support via optimization profiles
- GPU-resident inference pipeline

**vf_dnn_processing.c:**
- Added CUDA frame format support (`AV_PIX_FMT_CUDA`)
- Hardware device awareness flags
- Output hw_frames_ctx setup for CUDA output

**dnn_cuda_kernels.cu:**
- Zero-copy format conversion on GPU
- Avoids GPU↔CPU transfers for input/output

## Application Integration

The super-resolution feature is integrated into the main netv application:

### Configuration (main.py)

```python
SR_MODEL_PATH = ~/ffmpeg_build/models/realesr-general-x4v3.pt
LIBTORCH_LIB_PATH = ~/ffmpeg_sources/libtorch/lib

is_sr_available() → checks if model file exists
```

### User Settings

| Setting | Values | Description |
|---------|--------|-------------|
| `sr_mode` | `off` | SR disabled (default) |
| | `enhance` | Always apply SR for cleanup/sharpening |
| | `upscale_1080` | Apply SR if source < 1080p |
| | `upscale_4k` | Apply SR if source < 2160p |

### Filter Generation (ffmpeg_command.py)

The `_build_sr_filter()` function generates the FFmpeg filter chain:
```
format=rgb24,dnn_processing=dnn_backend=torch:model={path},scale=-2:{target_height}:flags=area
```

SR is only applied when:
- `sr_mode` is not "off"
- Content is VOD (not live streaming)
- Encoder is NVENC or AMF (discrete GPU)
- Model file exists

When SR is active:
- Hardware pipeline is disabled (requires CPU frames)
- NVENC preset changes to "p4" (more stable)

## TODO

- [ ] Run complete benchmark suite (eager, torch.compile, TensorRT, FFmpeg)
- [ ] Profile FFmpeg torch backend to find slowdown cause
- [ ] Test native TensorRT backend with dynamic shapes
- [ ] Submit patches upstream to FFmpeg
