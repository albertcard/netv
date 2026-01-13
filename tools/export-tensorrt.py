#!/usr/bin/env python3
"""Export PyTorch models to TensorRT engines for FFmpeg dnn_processing filter.

This script converts Real-ESRGAN and similar super-resolution models to TensorRT
engines (.engine files) that can be loaded by FFmpeg's TensorRT DNN backend.

Usage:
    # Export for 720p input (1280x720 -> 5120x2880 with 4x upscaling)
    python export-tensorrt.py --width 1280 --height 720 --output model_720p.engine

    # Export for 1080p input with FP16 precision
    python export-tensorrt.py --width 1920 --height 1080 --fp16 --output model_1080p_fp16.engine

    # Export from custom model
    python export-tensorrt.py --model /path/to/model.pth --width 1280 --height 720

Requirements:
    pip install torch onnx tensorrt

Example FFmpeg usage after export:
    ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model_720p.engine" output.mp4
"""

import argparse
import os
import sys
import tempfile

def get_model(model_path=None):
    """Load or download Real-ESRGAN model."""
    import torch

    if model_path:
        # Try loading as TorchScript first
        try:
            model = torch.jit.load(model_path, map_location='cpu')
            print(f"Loaded TorchScript model from {model_path}")
            return model, True
        except:
            pass

        # Try loading as state dict with architecture
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            state_dict = torch.load(model_path, map_location='cpu')
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded RRDBNet model from {model_path}")
            return model, False
        except ImportError:
            pass

        # Try SRVGGNetCompact (Real-ESRGAN-anime/general models)
        try:
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            state_dict = torch.load(model_path, map_location='cpu')
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded SRVGGNetCompact model from {model_path}")
            return model, False
        except ImportError:
            pass

        raise RuntimeError(f"Could not load model from {model_path}. Install basicsr or realesrgan package.")

    # Download default model from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="ai-forever/Real-ESRGAN",
        filename="RealESRGAN_x4.pth"
    )
    return get_model(model_path)


def export_onnx(model, width, height, onnx_path, is_torchscript=False):
    """Export model to ONNX format."""
    import torch

    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Input shape: 1x3x{height}x{width}")

    dummy_input = torch.randn(1, 3, height, width, device='cpu')

    if is_torchscript:
        # TorchScript models need different handling
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=17,
            do_constant_folding=True,
        )
    else:
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,  # Fixed shape for best TRT performance
        )

    print(f"  ONNX export complete")


def build_engine(onnx_path, engine_path, fp16=False, workspace_gb=4):
    """Build TensorRT engine from ONNX model."""
    import tensorrt as trt

    print(f"Building TensorRT engine: {engine_path}")
    print(f"  FP16: {fp16}")
    print(f"  Workspace: {workspace_gb} GB")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  Warning: FP16 not supported on this platform")

    # Build engine
    print("  Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"  Engine saved: {engine_path} ({os.path.getsize(engine_path) / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to TensorRT engines')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to PyTorch model (.pth or .pt). Downloads Real-ESRGAN if not specified.')
    parser.add_argument('--width', '-W', type=int, default=1280,
                        help='Input width (default: 1280)')
    parser.add_argument('--height', '-H', type=int, default=720,
                        help='Input height (default: 720)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output engine path. Default: model_<width>x<height>[_fp16].engine')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision (recommended for best performance)')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size in GB (default: 4)')
    parser.add_argument('--keep-onnx', action='store_true',
                        help='Keep intermediate ONNX file')
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        suffix = '_fp16' if args.fp16 else ''
        args.output = f"model_{args.width}x{args.height}{suffix}.engine"

    # Load model
    print("=" * 60)
    print("Real-ESRGAN to TensorRT Export")
    print("=" * 60)
    model, is_torchscript = get_model(args.model)

    # Export to ONNX
    if args.keep_onnx:
        onnx_path = args.output.replace('.engine', '.onnx')
    else:
        fd, onnx_path = tempfile.mkstemp(suffix='.onnx')
        os.close(fd)

    try:
        export_onnx(model, args.width, args.height, onnx_path, is_torchscript)

        # Build TensorRT engine
        build_engine(onnx_path, args.output, fp16=args.fp16, workspace_gb=args.workspace)
    finally:
        if not args.keep_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)

    print()
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)
    print()
    print("Usage with FFmpeg:")
    print(f'  ffmpeg -i input.mp4 -vf "scale={args.width}:{args.height},dnn_processing=dnn_backend=tensorrt:model={args.output}" output.mp4')
    print()
    print("Note: Input video must be exactly {width}x{height} for this engine.".format(
        width=args.width, height=args.height))


if __name__ == '__main__':
    main()
