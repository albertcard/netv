#!/usr/bin/env python3
"""Export PyTorch models to TensorRT engines for FFmpeg dnn_processing filter.

This script converts Real-ESRGAN and similar super-resolution models to TensorRT
engines (.engine files) that can be loaded by FFmpeg's TensorRT DNN backend.

Supports dynamic input shapes - a single engine handles a range of resolutions.

Usage:
    # Export with dynamic shapes (default: 270p to 1280p)
    python export-tensorrt.py --output model.engine

    # Export with custom height range
    python export-tensorrt.py --min-height 360 --max-height 1080

    # Export from custom model
    python export-tensorrt.py --model /path/to/model.pth

Requirements:
    pip install torch onnx tensorrt

Example FFmpeg usage after export:
    ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model.engine" output.mp4
"""

import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDBNet."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture for Real-ESRGAN."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # Upsample 4x
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def get_model(model_path=None):
    """Load or download Real-ESRGAN model."""
    if model_path is None:
        # Download from HuggingFace
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
        print(f"Downloaded model to {model_path}")

    # Load weights
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']

    # Create model and load weights
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded RRDBNet model ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


def export_onnx(model, opt_shape, onnx_path):
    """Export model to ONNX format with dynamic axes."""
    opt_w, opt_h = opt_shape
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Optimal shape: 1x3x{opt_h}x{opt_w}")

    dummy_input = torch.randn(1, 3, opt_h, opt_w, device='cpu')

    # Dynamic axes for height and width (dimensions 2 and 3)
    dynamic_axes = {
        'input': {2: 'height', 3: 'width'},
        'output': {2: 'out_height', 3: 'out_width'}
    }

    # Use legacy exporter to embed weights in ONNX file (dynamo exports to separate .data file)
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy exporter to embed weights
    )

    print(f"  ONNX export complete (dynamic H/W)")


def build_engine(onnx_path, engine_path, min_shape, opt_shape, max_shape, fp16=False, workspace_gb=4):
    """Build TensorRT engine from ONNX model with dynamic shapes."""
    import tensorrt as trt

    min_w, min_h = min_shape
    opt_w, opt_h = opt_shape
    max_w, max_h = max_shape

    print(f"Building TensorRT engine: {engine_path}")
    print(f"  Dynamic shapes:")
    print(f"    min: {min_w}x{min_h}")
    print(f"    opt: {opt_w}x{opt_h}")
    print(f"    max: {max_w}x{max_h}")
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

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    # Shape format: (batch, channels, height, width)
    profile.set_shape(input_name,
                      min=(1, 3, min_h, min_w),
                      opt=(1, 3, opt_h, opt_w),
                      max=(1, 3, max_h, max_w))
    config.add_optimization_profile(profile)

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


def height_to_shape(h, aspect=16/9):
    """Convert height to (width, height) assuming aspect ratio."""
    w = int(h * aspect)
    # Round width to multiple of 8 for GPU alignment
    w = (w + 7) // 8 * 8
    return (w, h)


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to TensorRT engines')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to PyTorch model (.pth or .pt). Downloads Real-ESRGAN if not specified.')
    parser.add_argument('--min-height', type=int, default=270,
                        help='Minimum input height (default: 270)')
    parser.add_argument('--opt-height', type=int, default=720,
                        help='Optimal input height (default: 720)')
    parser.add_argument('--max-height', type=int, default=1280,
                        help='Maximum input height (default: 1280)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output engine path. Default: realesrgan_dynamic_fp16.engine')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 precision (default: enabled)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision instead of FP16')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size in GB (default: 4)')
    parser.add_argument('--keep-onnx', action='store_true',
                        help='Keep intermediate ONNX file')
    args = parser.parse_args()

    # Handle fp32 flag
    if args.fp32:
        args.fp16 = False

    # Convert heights to (width, height) shapes assuming 16:9
    # Ensure min <= opt <= max
    min_h = min(args.min_height, args.max_height)
    max_h = args.max_height
    opt_h = min(max(args.opt_height, min_h), max_h)
    min_shape = height_to_shape(min_h)
    opt_shape = height_to_shape(opt_h)
    max_shape = height_to_shape(max_h)

    # Set output path
    if args.output is None:
        suffix = '_fp16' if args.fp16 else '_fp32'
        args.output = f"realesrgan_dynamic{suffix}.engine"

    # Load model
    print("=" * 60)
    print("Real-ESRGAN to TensorRT Export (Dynamic Shapes)")
    print("=" * 60)
    model = get_model(args.model)

    # Export to ONNX
    if args.keep_onnx:
        onnx_path = args.output.replace('.engine', '.onnx')
    else:
        fd, onnx_path = tempfile.mkstemp(suffix='.onnx')
        os.close(fd)

    try:
        export_onnx(model, opt_shape, onnx_path)

        # Build TensorRT engine
        build_engine(onnx_path, args.output,
                     min_shape=min_shape,
                     opt_shape=opt_shape,
                     max_shape=max_shape,
                     fp16=args.fp16,
                     workspace_gb=args.workspace)
    finally:
        if not args.keep_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)

    print()
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)
    print()
    print(f"Engine accepts input heights from {args.min_height} to {args.max_height} (16:9)")
    print()
    print("Usage with FFmpeg:")
    print(f'  ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model={args.output}" output.mp4')


if __name__ == '__main__':
    main()
