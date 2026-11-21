"""Перевірка гіпотези: чому FP32 на RTX 2050 настільки повільніший за FP16 для EfficientNet-B0.

Частина 1 — пошарово: кожен Conv2d мережі з його реальною формою входу (batch 64)
окремо проганяється forward+backward у FP32 (TF32 off) і FP16; час підсумовується
за типами: depthwise (groups = in_channels), pointwise 1×1, звичайні k×k.
Частина 2 — torch.profiler на повному кроці навчання: топ CUDA-ядер за часом.

Запуск: python scripts/profile_precision.py
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.model import build_model  # noqa: E402

BS = 32  # при 64 FP32 не влазить у 4 ГБ VRAM (5.5 ГБ) — тоді міряли б не обчислення, а підкачку через PCIe
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def conv_kind(c: nn.Conv2d) -> str:
    if c.groups > 1 and c.groups == c.in_channels:
        return "depthwise k×k"
    if c.kernel_size == (1, 1):
        return "pointwise 1×1"
    return "звичайна k×k"


def capture_conv_shapes():
    m = build_model("efficientnet_b0", 17, pretrained=False).cuda().eval()
    shapes = []
    hooks = [mod.register_forward_hook(lambda mod, inp, out: shapes.append((mod, tuple(inp[0].shape))))
             for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    with torch.no_grad():
        m(torch.randn(BS, 3, 224, 224, device="cuda"))
    for h in hooks:
        h.remove()
    return shapes


def time_layer(conv, in_shape, dtype, iters=5, warmup=3):
    c = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                  groups=conv.groups, bias=conv.bias is not None).cuda().to(dtype)
    x = torch.randn(in_shape, device="cuda", dtype=dtype, requires_grad=True)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        y = c(x)
        y.backward(torch.ones_like(y))
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms forward+backward


def part1():
    shapes = capture_conv_shapes()
    totals = {torch.float32: defaultdict(float), torch.float16: defaultdict(float)}
    counts = defaultdict(int)
    for conv, shp in shapes:
        k = conv_kind(conv)
        counts[k] += 1
        for dt in (torch.float32, torch.float16):
            totals[dt][k] += time_layer(conv, shp, dt)
    print(f"\nЧастина 1. Сумарний час forward+backward усіх Conv2d (batch {BS}), мс:")
    print(f"  {'тип':16s} {'к-сть':>5s} {'FP32':>10s} {'FP16':>10s} {'FP32/FP16':>10s} {'частка FP32':>12s}")
    s32 = sum(totals[torch.float32].values()); s16 = sum(totals[torch.float16].values())
    for k in sorted(counts):
        a, b = totals[torch.float32][k], totals[torch.float16][k]
        print(f"  {k:16s} {counts[k]:5d} {a:10.1f} {b:10.1f} {a/b:10.1f}× {a/s32:11.1%}")
    print(f"  {'УСІ Conv2d':16s} {len(shapes):5d} {s32:10.1f} {s16:10.1f} {s32/s16:10.1f}×")
    return totals, counts


def part2():
    print("\nЧастина 2. torch.profiler, 1 крок навчання (після прогріву), топ-8 CUDA-ядер:")
    for amp in (False, True):
        m = build_model("efficientnet_b0", 17, pretrained=False).cuda().train()
        opt = torch.optim.AdamW(m.parameters())
        sc = torch.amp.GradScaler("cuda", enabled=amp)
        x = torch.randn(BS, 3, 224, 224, device="cuda"); y = torch.randint(0, 17, (BS,), device="cuda")

        def step():
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = nn.functional.cross_entropy(m(x), y)
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step(); torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
        total = sum(e.self_device_time_total for e in evs)
        evs.sort(key=lambda e: e.self_device_time_total, reverse=True)
        print(f"\n  {'AMP FP16' if amp else 'FP32 (TF32 off)'}: сумарний час CUDA-ядер {total/1000:.1f} мс")
        for e in evs[:8]:
            print(f"    {e.self_device_time_total/1000:8.1f} мс {e.self_device_time_total/total:6.1%}  "
                  f"x{e.count:<4d} {e.key[:95]}")
        del m, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | cuDNN {torch.backends.cudnn.version()}"
          f" | cudnn.benchmark=True | TF32 off")
    part1()
    part2()
