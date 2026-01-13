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
| `tools/download-sr-models.sh` | Download .pth, convert to TorchScript (.pt) | Ready |
| `tools/compile-sr-tensorrt.sh` | Compile .pt → TensorRT (-trt.pt) | Ready |
| `tools/test-realesr.sh` | Benchmark FFmpeg + libtorch | Ready |
| `tools/install-ffmpeg.sh` | Build FFmpeg with libtorch support | Ready |

## FFmpeg Patches

Location: `tools/patches/`

| File | Type | Purpose |
|------|------|---------|
| `ffmpeg-dnn-cuda-frames.patch` | Unified patch | CUDA zero-copy for dnn_backend_torch + vf_dnn_processing |
| `patch_cuda_dlopen.py` | Python script | Adds dlopen for libtorch_cuda.so |
| `patch_dnn_cuda_frames.py` | Python script | CUDA frame support |
| `patch_dnn_cuda_frames_v2.py` | Python script | Updated version |
| `patch_torch_input_cuda.py` | Python script | CUDA input handling |
| `patch_torch_zerocopy_full.py` | Python script | Full zero-copy implementation |
| `patch_zerocopy.py` | Python script | Basic zero-copy |

### Main patch (ffmpeg-dnn-cuda-frames.patch)

Adds CUDA frame support to avoid GPU↔CPU copies:
- Modifies `libavfilter/dnn/dnn_backend_torch.cpp`
- Modifies `libavfilter/vf_dnn_processing.c`
- Creates tensors directly from CUDA device pointers
- Enables pipeline: `hwupload_cuda -> dnn_processing -> nvenc`

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

### Download and convert models
```bash
cd ~/projects/netv
./tools/download-sr-models.sh
```

### Compile TensorRT model (optional, for faster inference)
```bash
MODEL=RealESRGAN_x4plus.pt ./tools/compile-sr-tensorrt.sh
```

### Test FFmpeg performance
```bash
./tools/test-realesr.sh
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

**Export model to TensorRT engine:**
```bash
# Export for 720p input (creates model_1280x720_fp16.engine)
python tools/export-tensorrt.py --width 1280 --height 720 --fp16

# Export for 1080p input
python tools/export-tensorrt.py --width 1920 --height 1080 --fp16
```

**Run with FFmpeg:**
```bash
ffmpeg -i input_720p.mp4 \
  -vf "dnn_processing=dnn_backend=tensorrt:model=model_1280x720_fp16.engine" \
  -c:v hevc_nvenc output.mp4
```

**Note:** TensorRT engines are compiled for fixed input dimensions. You need separate
engines for different resolutions (720p, 1080p, etc.). Use `scale=W:H` before
dnn_processing to match the engine's expected input.

### Option B: LibTorch + torch_tensorrt

Alternative approach using libtorch with TensorRT-compiled TorchScript models.
Requires larger dependencies but supports dynamic shapes.

**Build FFmpeg with LibTorch:**
```bash
ENABLE_TENSORRT=0 ENABLE_LIBTORCH=1 ./tools/install-ffmpeg.sh
```

**Compile model with torch_tensorrt:**

```bash
./tools/compile-sr-tensorrt.sh
```

Creates `realesr-general-x4v3-trt.pt` optimized for FP16 inference.

**Run TensorRT model with FFmpeg:**

```bash
TRT_LIB=~/ffmpeg_build/models/.venv/lib/python3.12/site-packages/torch_tensorrt/lib
TRTCORE=~/ffmpeg_build/models/.venv/lib/python3.12/site-packages/tensorrt_libs
TORCH_LIB=~/.local/lib

LD_LIBRARY_PATH="$TRT_LIB:$TRTCORE:$TORCH_LIB:$LD_LIBRARY_PATH" \
ffmpeg -i input.mp4 \
  -vf "dnn_processing=dnn_backend=torch:model=$HOME/ffmpeg_build/models/realesr-general-x4v3-trt.pt:device=cuda" \
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

## What's in the patches vs what's in ffmpeg source

**PATCHES ARE NOT UP TO DATE WITH FFMPEG SOURCE.**

Full diffs saved to `tools/patches/`:
- `dnn_backend_torch.diff` (489 lines) - current source vs clean tarball
- `vf_dnn_processing.diff` (125 lines) - current source vs clean tarball

| Item | `ffmpeg-dnn-cuda-frames.patch` | Current ffmpeg source |
|------|--------------------------------|----------------------|
| Color conversion | PyTorch tensor ops only | NPP (`nppi_color_conversion.h`) |
| NPP includes | No | Yes |
| NPP `_Ctx` API | No | Yes (for CUDA 13+) |
| Worker thread code | Preserved from upstream | **Removed** |
| Complexity | ~173 lines added | ~489 lines changed |

### Key differences

**1. Worker thread removal:** The current source removes the worker thread infrastructure from upstream ffmpeg (the `worker_thread`, `mutex`, `cond`, `pending_queue`, `worker_stop` fields). The patch file preserves the upstream structure.

**2. NPP color conversion:** The current source uses NPP's `nppiNV12ToRGB_8u_P2C3R_Ctx` for NV12→RGB conversion. The patch file uses PyTorch tensor ops for all color conversion.

**3. Output path:** The current source uses PyTorch tensor ops for RGB→NV12 on output (BT.601 matrix, avg_pool2d for chroma downsampling). The patch file copies tensor data directly to CUDA frame.

### Recommendation

Revert ffmpeg to clean state, apply existing `ffmpeg-dnn-cuda-frames.patch`, benchmark first. The NPP changes were unauthorized scope creep and haven't been proven to help the 4x performance gap. The bottleneck is likely elsewhere (JIT recompilation, CUDA context switching, per-frame sync).

## Files modified (not in patches)

These changes exist in `~/ffmpeg_sources/ffmpeg-snapshot/` but are NOT in the patch files:

1. `libavfilter/dnn/dnn_backend_torch.cpp` (489 line diff):
   - NPP includes (`nppi_color_conversion.h`, `cuda_runtime.h`)
   - NPP `_Ctx` API calls for CUDA 13+ compatibility
   - NppStreamContext setup per frame
   - Removed worker thread infrastructure
   - Full NV12↔RGB conversion via NPP and PyTorch

2. `libavfilter/vf_dnn_processing.c` (125 line diff):
   - Added `hw_frames_ctx` field
   - Added `AV_PIX_FMT_CUDA` to supported formats
   - Output hw_frames_ctx setup for CUDA frames
   - AVFILTER_FLAG_HWDEVICE and FF_FILTER_FLAG_HWFRAME_AWARE

3. `ffbuild/config.mak`:
   - Added `-lnppicc -lnppc -lcudart` to EXTRALIBS

## Next Steps

1. **Revert ffmpeg to clean state:**
   ```bash
   cd ~/ffmpeg_sources
   rm -rf ffmpeg-snapshot
   tar xjf ffmpeg-snapshot.tar.bz2
   # Or re-download: curl -O https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
   ```

2. **Apply patches:**
   ```bash
   cd ~/ffmpeg_sources/ffmpeg-snapshot
   patch -p1 < ~/projects/netv/tools/patches/ffmpeg-dnn-cuda-frames.patch
   ```

3. **Rebuild ffmpeg:**
   ```bash
   cd ~/projects/netv
   ./tools/install-ffmpeg.sh
   ```

4. **Run benchmark suite** with RealESRGAN_x4plus at 1280x720

5. **Profile** to find the 4x slowdown cause

6. **Optimize** based on actual data

## TODO

- [ ] Revert ffmpeg source to clean state
- [ ] Apply patches from `tools/patches/`
- [ ] Run complete benchmark suite (eager, torch.compile, TensorRT, FFmpeg)
- [ ] Profile FFmpeg torch backend to find slowdown cause
- [ ] Test TensorRT integration via FFmpeg
- [ ] Submit patches upstream to FFmpeg
