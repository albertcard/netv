#!/bin/bash
# Benchmark FFmpeg super-resolution performance
set -e

export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/home/jvdillon/.local/lib/libtorch_cuda.so

FFMPEG=/home/jvdillon/.local/bin/ffmpeg
MODEL_DIR=/home/jvdillon/ffmpeg_build/models

MODEL=${MODEL:-realesr-general-x4v3.pt}
RESOLUTION=${RESOLUTION:-1280x720}
FRAMES=${FRAMES:-30}
DEVICE=${DEVICE:-cuda}

MODEL_PATH="${MODEL_DIR}/${MODEL}"

echo "=== FFmpeg Super-Resolution Benchmark ==="
echo "Model: $MODEL"
echo "Resolution: $RESOLUTION"
echo "Frames: $FRAMES"
echo "Device: $DEVICE"
echo ""

# Run benchmark
$FFMPEG -benchmark -hide_banner -y \
  -f lavfi -i "testsrc=duration=10:size=${RESOLUTION}:rate=30" \
  -vf "format=rgb24,dnn_processing=dnn_backend=torch:model=${MODEL_PATH}:device=${DEVICE}" \
  -frames:v "$FRAMES" \
  -f null /dev/null 2>&1 | tee /tmp/sr-benchmark.log

echo ""
echo "=== Results ==="
grep -E "frame=|speed=|fps=" /tmp/sr-benchmark.log | tail -1
grep -E "utime=|stime=|rtime=" /tmp/sr-benchmark.log || true
