#!/bin/bash
# Install super-resolution model for FFmpeg TensorRT backend
# Downloads realesr-general-x4v3 (compact model) and builds TensorRT engine
#
# TODO: Harvest additional models from https://openmodeldb.info/
#       - Browse by architecture (Compact, ESRGAN, etc.) and scale factor
#       - Many community-trained models optimized for specific content types
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"

echo "========================================"
echo "Super-Resolution Model Installation"
echo "========================================"
echo "Model: realesr-general-x4v3 (SRVGGNetCompact)"
echo "  - 1.2M params, ~64 fps @ 720p->4K"
echo "  - 4x upscale"
echo ""
echo "Output: $MODEL_DIR/"
echo ""

# Create output directory
mkdir -p "$MODEL_DIR"

# Setup Python venv
VENV_DIR="$MODEL_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch onnx tensorrt==10.14.1.48.post1

# Build engines for common resolutions (FFmpeg TensorRT backend needs fixed shapes)
echo ""
echo "Building TensorRT engines for common resolutions..."

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

deactivate

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
