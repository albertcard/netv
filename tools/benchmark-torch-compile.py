#!/usr/bin/env python3
"""Benchmark torch.compile() modes for Real-ESRGAN inference."""
import torch
import time
import argparse

def benchmark(model, x, frames, name):
    """Run benchmark and return fps."""
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(frames):
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fps = frames / elapsed
    ms = 1000 * elapsed / frames
    print(f"  {name}: {fps:.1f} fps ({ms:.1f} ms/frame)")
    return fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/jvdillon/ffmpeg_build/models/realesr-general-x4v3.pt')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--frames', type=int, default=100)
    args = parser.parse_args()

    print(f"=== torch.compile() Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Input: {args.width}x{args.height}")
    print(f"Output: {args.width*4}x{args.height*4}")
    print(f"Frames: {args.frames}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Load model
    model = torch.jit.load(args.model).cuda().eval()
    x = torch.rand(1, 3, args.height, args.width, device='cuda')

    results = {}

    # 1. Baseline: JIT model (no compile)
    print("1. TorchScript JIT (baseline):")
    results['jit'] = benchmark(model, x, args.frames, "JIT")

    # 2. torch.compile() with default mode
    print("\n2. torch.compile(mode='default'):")
    try:
        model_default = torch.compile(model, mode='default')
        # First call triggers compilation
        with torch.no_grad():
            _ = model_default(x)
        torch.cuda.synchronize()
        results['compile_default'] = benchmark(model_default, x, args.frames, "default")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['compile_default'] = None

    # 3. torch.compile() with reduce-overhead mode
    print("\n3. torch.compile(mode='reduce-overhead'):")
    try:
        model_reduce = torch.compile(model, mode='reduce-overhead')
        with torch.no_grad():
            _ = model_reduce(x)
        torch.cuda.synchronize()
        results['compile_reduce'] = benchmark(model_reduce, x, args.frames, "reduce-overhead")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['compile_reduce'] = None

    # 4. torch.compile() with max-autotune mode
    print("\n4. torch.compile(mode='max-autotune'):")
    try:
        model_autotune = torch.compile(model, mode='max-autotune')
        with torch.no_grad():
            _ = model_autotune(x)
        torch.cuda.synchronize()
        results['compile_autotune'] = benchmark(model_autotune, x, args.frames, "max-autotune")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['compile_autotune'] = None

    # 5. torch.compile() with inductor backend explicitly
    print("\n5. torch.compile(backend='inductor', mode='max-autotune'):")
    try:
        model_inductor = torch.compile(model, backend='inductor', mode='max-autotune')
        with torch.no_grad():
            _ = model_inductor(x)
        torch.cuda.synchronize()
        results['inductor'] = benchmark(model_inductor, x, args.frames, "inductor+autotune")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['inductor'] = None

    # Summary
    print("\n=== Summary ===")
    baseline = results['jit']
    for name, fps in results.items():
        if fps:
            speedup = fps / baseline
            print(f"{name:20s}: {fps:5.1f} fps ({speedup:.2f}x vs JIT)")
        else:
            print(f"{name:20s}: FAILED")

if __name__ == '__main__':
    main()
