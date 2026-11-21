"""Бенчмарк точності обчислень: FP32 (TF32 off / on) vs AMP FP16 на кроці навчання EfficientNet-B0.

Методика:
- синтетичний батч 64×3×224×224 (вимірюємо лише GPU, без DataLoader);
- однаковий torch.backends.cudnn.benchmark для всіх режимів (за замовчуванням True, як у train.py);
- TF32 задається ЯВНО для кожного режиму (cudnn.allow_tf32 і cuda.matmul.allow_tf32);
- для кожного режиму: нова модель, прогрів (WARMUP кроків, у т.ч. автотюнінг cudnn), потім STEPS кроків;
- 3 повтори, режими чергуються по колу (r1: A,B,C; r2: A,B,C; …), щоб нагрів/частоти не давали перевагу одному режиму.

Запуск: python scripts/bench_precision.py [--benchmark 0|1] [--repeats 3]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.model import build_model  # noqa: E402

MODES = {
    # name: (amp, allow_tf32)
    "FP32 (TF32 off)": (False, False),
    "FP32 (TF32 on)": (False, True),
    "AMP FP16": (True, False),
}


def train_step_bench(amp, tf32, bs, warmup, steps, arch="efficientnet_b0"):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.manual_seed(0)
    m = build_model(arch, 17, pretrained=False).cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randn(bs, 3, 224, 224, device="cuda")
    y = torch.randint(0, 17, (bs,), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 2**20
    del m, opt
    torch.cuda.empty_cache()
    return steps * bs / dt, dt / steps * 1000, vram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=int, default=1, help="torch.backends.cudnn.benchmark (однаково для всіх)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = bool(args.benchmark)
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | cuDNN {torch.backends.cudnn.version()}")
    print(f"Defaults before override: cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
          f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"cudnn.benchmark={torch.backends.cudnn.benchmark} | batch {args.batch_size} | "
          f"warmup {args.warmup} | steps {args.steps} | repeats {args.repeats}", flush=True)

    res = {k: [] for k in MODES}
    for r in range(args.repeats):
        for name, (amp, tf32) in MODES.items():
            ips, ms, vram = train_step_bench(amp, tf32, args.batch_size, args.warmup, args.steps)
            res[name].append({"img_per_s": ips, "ms_per_step": ms, "peak_vram_mb": vram})
            print(f"  repeat {r+1} | {name:16s} | cudnn.allow_tf32={tf32!s:5s} matmul.allow_tf32={tf32!s:5s} | "
                  f"{ips:7.1f} img/s | {ms:8.1f} ms/step | VRAM {vram:5.0f} MB", flush=True)

    print("\nПідсумок (медіана з повторів):")
    summary = {}
    base = statistics.median(x["img_per_s"] for x in res["FP32 (TF32 off)"])
    for name, runs in res.items():
        ips = [x["img_per_s"] for x in runs]
        med = statistics.median(ips)
        summary[name] = {"median_img_per_s": med, "min": min(ips), "max": max(ips),
                         "median_ms_per_step": statistics.median(x["ms_per_step"] for x in runs),
                         "peak_vram_mb": max(x["peak_vram_mb"] for x in runs), "speedup_vs_fp32": med / base}
        print(f"  {name:16s}: {med:7.1f} img/s (min {min(ips):.1f}, max {max(ips):.1f}) "
              f"→ ×{med/base:.2f} відносно FP32 (TF32 off)")
    if args.json:
        args.json.write_text(json.dumps({"config": vars(args) | {"json": str(args.json)},
                                         "runs": res, "summary": summary}, indent=2, ensure_ascii=False),
                             encoding="utf-8")


if __name__ == "__main__":
    main()
