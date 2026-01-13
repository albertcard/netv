#!/bin/bash
# Build TensorRT engines for AI Upscale (Real-ESRGAN)
#
# Prerequisites: uv sync --group ai_upscale
#   Or: pip install torch onnx tensorrt
#
# TODO: Harvest additional models from https://openmodeldb.info/
#       - Browse by architecture (Compact, ESRGAN, etc.) and scale factor
#       - Many community-trained models optimized for specific content types
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"

echo "========================================"
echo "AI Upscale: TensorRT Engine Builder"
echo "========================================"
echo "Model: realesr-general-x4v3 (SRVGGNetCompact)"
echo "  - 1.2M params, ~64 fps @ 720p->4K"
echo "  - 4x upscale"
echo ""
echo "Output: $MODEL_DIR/"
echo ""

# Check dependencies
if ! python3 -c "import torch, onnx, tensorrt" 2>/dev/null; then
    echo "ERROR: Missing dependencies. Install with:"
    echo "  uv sync --group ai_upscale"
    echo "Or:"
    echo "  pip install torch onnx tensorrt"
    exit 1
fi

# Create output directory
mkdir -p "$MODEL_DIR"

# Build engines for common resolutions (FFmpeg TensorRT backend needs fixed shapes)
echo "Building TensorRT engines for common resolutions..."
echo ""

for res in 480 720 1080; do
    engine="$MODEL_DIR/realesrgan_${res}p_fp16.engine"
    if [ -f "$engine" ]; then
        echo "  ${res}p: already exists, skipping"
    else
        echo "  ${res}p: building..."
        python3 "$SCRIPT_DIR/export-tensorrt.py" \
            --model-type compact \
            --min-height $res --opt-height $res --max-height $res \
            -o "$engine" 2>&1 | grep -E "^(Downloading|Loading|Loaded|Engine saved)"
    fi
done

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Engines:"
ls -lh "$MODEL_DIR"/*.engine 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""
echo "Test with:"
echo "  ffmpeg -init_hw_device cuda=cu -filter_hw_device cu \\"
echo "    -f lavfi -i testsrc=duration=3:size=1280x720:rate=30 \\"
echo "    -vf \"format=rgb24,hwupload,dnn_processing=dnn_backend=8:model=$MODEL_DIR/realesrgan_720p_fp16.engine\" \\"
echo "    -c:v hevc_nvenc test.mp4"
