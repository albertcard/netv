#!/bin/bash
# Test Real-ESRGAN super-resolution performance via ffmpeg libtorch backend
set -e

FFMPEG="${FFMPEG:-$HOME/ffmpeg_build/bin/ffmpeg}"
MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
MODEL="${MODEL:-realesr-general-x4v3.pt}"
MODEL_PATH="$MODEL_DIR/$MODEL"

# Test parameters
DURATION=${DURATION:-5}
RESOLUTION=${RESOLUTION:-1280x720}
FPS=${FPS:-30}
FRAMES=$((DURATION * FPS))

echo "=== Real-ESRGAN Performance Test ==="
echo ""

# Check ffmpeg
if [ ! -x "$FFMPEG" ]; then
    echo "ERROR: ffmpeg not found at $FFMPEG"
    echo "Run ./tools/install-ffmpeg.sh first"
    exit 1
fi

# Check libtorch support
if ! "$FFMPEG" -hide_banner -filters 2>/dev/null | grep -q dnn_processing; then
    echo "ERROR: ffmpeg not built with dnn_processing filter"
    echo "Rebuild with ENABLE_TORCH=1"
    exit 1
fi

# Check model
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Run ./tools/download-sr-models.sh first"
    exit 1
fi

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    echo "GPU: Not detected (nvidia-smi not found)"
fi

echo "FFmpeg: $FFMPEG"
echo "Model: $MODEL_PATH"
echo "Test: ${RESOLUTION} @ ${FPS}fps, ${FRAMES} frames"
echo ""

# Preload libtorch_cuda if needed (workaround for dynamic loading)
LIBTORCH_DIR="$HOME/ffmpeg_sources/libtorch/lib"
if [ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]; then
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so"
fi

run_test() {
    local device="$1"
    local label="$2"

    echo "--- $label ---"
    "$FFMPEG" -hide_banner -y \
        -f lavfi -i "testsrc=duration=${DURATION}:size=${RESOLUTION}:rate=${FPS}" \
        -vf "format=rgb24,dnn_processing=dnn_backend=torch:model=${MODEL_PATH}:device=${device}" \
        -frames:v "$FRAMES" -f null - 2>&1 | \
        grep -E "^frame=.*speed=" | tail -1 | \
        sed 's/.*fps=\([0-9.]*\).*/FPS: \1/' | head -1

    # Also show the raw speed multiplier
    "$FFMPEG" -hide_banner -y \
        -f lavfi -i "testsrc=duration=${DURATION}:size=${RESOLUTION}:rate=${FPS}" \
        -vf "format=rgb24,dnn_processing=dnn_backend=torch:model=${MODEL_PATH}:device=${device}" \
        -frames:v "$FRAMES" -f null - 2>&1 | \
        grep -E "^frame=.*speed=" | tail -1 | \
        sed 's/.*speed=\([0-9.]*\)x.*/Speed: \1x realtime/' | head -1
    echo ""
}

# Run CUDA test
run_test "cuda" "CUDA (GPU)"

# Optionally run CPU test for comparison
if [ "${TEST_CPU:-0}" = "1" ]; then
    unset LD_PRELOAD
    run_test "cpu" "CPU"
fi

echo "=== Done ==="
echo ""
echo "For real-time ${FPS}fps, need speed >= 1.0x"
echo "To also test CPU: TEST_CPU=1 $0"
