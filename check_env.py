"""Крок 1: перевірка середовища AgroScan (Python, PyTorch, CUDA, GPU)."""
import platform
import sys
import time

import torch
import torchvision

print(f"Python       : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
print(f"PyTorch      : {torch.__version__}")
print(f"torchvision  : {torchvision.__version__}")
print(f"CUDA (збірка): {torch.version.cuda}")
print(f"cuDNN        : {torch.backends.cudnn.version()}")

cuda_ok = torch.cuda.is_available()
print(f"CUDA доступна: {cuda_ok}")

if cuda_ok:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"GPU          : {torch.cuda.get_device_name(idx)}")
    print(f"VRAM         : {props.total_memory / 1024**3:.2f} ГБ")
    print(f"Compute cap. : {props.major}.{props.minor}")
    print(f"SM (мультипр.): {props.multi_processor_count}")

    # Короткий реальний тест: множення матриць 4096x4096 у FP32 і FP16 (AMP)
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    for dtype in (torch.float32, torch.float16):
        x, y = a.to(dtype), b.to(dtype)
        torch.matmul(x, y)  # прогрів
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            torch.matmul(x, y)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        tflops = 2 * 4096**3 / dt / 1e12
        print(f"matmul {str(dtype):14s}: {dt*1000:7.2f} мс  ≈ {tflops:.2f} TFLOPS")
else:
    print("УВАГА: GPU не знайдено — перевір драйвер і версію torch (+cuXXX).")
