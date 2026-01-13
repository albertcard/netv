#!/usr/bin/env python3
"""Benchmark pure PyTorch Real-ESRGAN inference to establish baseline."""
import torch
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/jvdillon/ffmpeg_build/models/realesr-general-x4v3.pt')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    print(f"=== PyTorch Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Input: {args.width}x{args.height}")
    print(f"Frames: {args.frames}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    model = torch.jit.load(args.model).cuda().eval()

    # Create input tensor (already on GPU)
    x = torch.rand(1, 3, args.height, args.width, device='cuda')

    # Warmup
    print(f"Warming up ({args.warmup} frames)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark - GPU only (best case)
    print("Benchmarking GPU-only (tensor stays on GPU)...")
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.frames):
            out = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fps_gpu = args.frames / elapsed
    print(f"  GPU-only: {fps_gpu:.1f} fps ({1000*elapsed/args.frames:.1f} ms/frame)")

    # Benchmark - with CPU copy (FFmpeg case)
    print("Benchmarking with CPU roundtrip (simulates FFmpeg)...")
    x_cpu = torch.rand(1, 3, args.height, args.width)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.frames):
            # Copy input to GPU
            x_gpu = x_cpu.cuda()
            # Run inference
            out_gpu = model(x_gpu)
            # Copy output to CPU (this is the expensive part!)
            out_cpu = out_gpu.cpu()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fps_cpu = args.frames / elapsed
    print(f"  With CPU copy: {fps_cpu:.1f} fps ({1000*elapsed/args.frames:.1f} ms/frame)")

    # Calculate overhead
    overhead = (1/fps_cpu - 1/fps_gpu) * 1000
    print()
    print(f"=== Results ===")
    print(f"GPU-only:      {fps_gpu:.1f} fps")
    print(f"With CPU copy: {fps_cpu:.1f} fps")
    print(f"Copy overhead: {overhead:.1f} ms/frame ({100*(fps_gpu-fps_cpu)/fps_gpu:.0f}% slowdown)")

    # Memory info
    out_size = out_gpu.numel() * out_gpu.element_size() / 1024 / 1024
    print(f"Output size:   {out_size:.1f} MB ({out_gpu.shape})")

if __name__ == '__main__':
    main()
