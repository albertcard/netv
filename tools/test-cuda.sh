#!/bin/bash
# Test FFmpeg CUDA inference
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/home/jvdillon/.local/lib/libtorch_cuda.so

MODEL=/home/jvdillon/ffmpeg_build/models/realesr-general-x4v3.pt
FFMPEG=/home/jvdillon/.local/bin/ffmpeg
FRAMES=${FRAMES:-2}
LOGLEVEL=${LOGLEVEL:-info}

echo "Testing FFmpeg CUDA inference..."
echo "Model: $MODEL"
echo "FFmpeg: $FFMPEG"
echo "Frames: $FRAMES"
echo ""

$FFMPEG -loglevel $LOGLEVEL -hide_banner -y \
  -f lavfi -i testsrc=duration=2:size=320x240:rate=5 \
  -vf format=rgb24,dnn_processing=dnn_backend=torch:model=${MODEL}:device=cuda \
  -frames:v $FRAMES \
  -f null -

echo ""
echo "Exit code: $?"
