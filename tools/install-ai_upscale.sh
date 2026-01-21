#!/bin/bash
# Build TensorRT engines for AI Upscale
#
# Prerequisites: uv sync --group ai_upscale
#   Or: pip install torch onnx tensorrt
#
# Models sourced from https://openmodeldb.info/
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
MODEL="${MODEL:-recommended}"

# Use uv run if in a uv project, otherwise plain python3
if [ -f "$PROJECT_DIR/pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
    PYTHON="uv run --project $PROJECT_DIR python3"
else
    PYTHON="python3"
fi

# Show help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [MODEL]"
    echo ""
    echo "Build TensorRT engines for AI Upscale."
    echo ""
    echo "Arguments:"
    echo "  MODEL    Model to build (default: $MODEL)"
    echo "           'recommended' - 4x-compact, 2x-liveaction-span"
    echo "           'all'         - all models including 4x-realesrgan"
    echo ""
    echo "Environment:"
    echo "  MODEL_DIR   Output directory (default: \$HOME/ffmpeg_build/models)"
    echo "  MODEL       Model name (can also be passed as argument)"
    echo ""
    echo "Available models:"
    $PYTHON "$SCRIPT_DIR/export-tensorrt.py" --list
    exit 0
fi

# Allow model to be passed as argument
if [ -n "$1" ]; then
    MODEL="$1"
fi

# Handle "recommended" option - build recommended models
if [ "$MODEL" = "recommended" ]; then
    echo "========================================"
    echo "AI Upscale: Building recommended models"
    echo "========================================"
    echo ""
    for m in 4x-compact 2x-liveaction-span; do
        echo ">>> Building $m..."
        MODEL="$m" "$0"
        echo ""
    done
    echo "Done! Recommended models built."
    exit 0
fi

# Handle "all" option - build all available models
if [ "$MODEL" = "all" ]; then
    echo "========================================"
    echo "AI Upscale: Building ALL models"
    echo "========================================"
    echo ""
    for m in 4x-compact 2x-liveaction-span 4x-realesrgan; do
        echo ">>> Building $m..."
        MODEL="$m" "$0"
        echo ""
    done
    echo "Done! All models built."
    exit 0
fi

echo "========================================"
echo "AI Upscale: TensorRT Engine Builder"
echo "========================================"
echo "Model: $MODEL"
echo "Output: $MODEL_DIR/"
echo ""

# Check dependencies
if ! $PYTHON -c "import torch, onnx, tensorrt" 2>/dev/null; then
    echo "ERROR: Missing dependencies. Install with:"
    echo "  uv sync --group ai_upscale"
    echo "Or:"
    echo "  pip install torch onnx tensorrt"
    exit 1
fi

# Create output directory
mkdir -p "$MODEL_DIR"

# Determine resolutions based on model scale
case "$MODEL" in
    2x-*)
        # 2x models: 720p and 1080p input -> 1440p and 4K output
        RESOLUTIONS="720 1080"
        ;;
    4x-*)
        # 4x models: 480p, 720p, 1080p input -> 1080p, 4K, 4K output
        RESOLUTIONS="480 720 1080"
        ;;
    *)
        # Default to 2x resolutions
        RESOLUTIONS="720 1080"
        ;;
esac

# Build engines for common resolutions (FFmpeg TensorRT backend needs fixed shapes)
echo "Building TensorRT engines for resolutions: $RESOLUTIONS"
echo ""

for res in $RESOLUTIONS; do
    engine="$MODEL_DIR/${MODEL}_${res}p_fp16.engine"
    if [ -f "$engine" ]; then
        echo "  ${res}p: already exists, skipping"
    else
        echo "  ${res}p: building..."
        # Capture output to show errors if build fails
        OUTPUT=$($PYTHON "$SCRIPT_DIR/export-tensorrt.py" \
            --model "$MODEL" \
            --min-height $res --opt-height $res --max-height $res \
            -o "$engine" 2>&1) || {
            echo "ERROR building ${res}p engine:"
            echo "$OUTPUT"
            exit 1
        }
        # Show filtered progress on success
        echo "$OUTPUT" | grep -E "^(Downloading|Using cached|Loading|Using ONNX|Engine saved|  )" || true
        # Verify engine was created
        if [ ! -f "$engine" ]; then
            echo "ERROR: Engine file not created: $engine"
            echo "Build output:"
            echo "$OUTPUT"
            exit 1
        fi
    fi
done

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Engines built:"
ls -lh "$MODEL_DIR"/${MODEL}_*.engine 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""
echo "To use a different model, run:"
echo "  MODEL=2x-liveaction-span $0"
echo "  MODEL=4x-compact $0"
echo ""
echo "Test with:"
echo "  ffmpeg -init_hw_device cuda=cu -filter_hw_device cu \\"
echo "    -f lavfi -i testsrc=duration=3:size=1920x1080:rate=30 \\"
echo "    -vf \"format=rgb24,hwupload,dnn_processing=dnn_backend=8:model=$MODEL_DIR/${MODEL}_1080p_fp16.engine\" \\"
echo "    -c:v hevc_nvenc test.mp4"
