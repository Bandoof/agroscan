"""Крок 4c: час інференсу натренованої моделі на CPU і GPU.

Вимірюємо лише модель (вхід 1×3×224×224 уже на пристрої), режим eval + inference_mode.
Прогрів WARMUP запусків, далі RUNS вимірювань → медіана і p95 (мс).
Окремо: пропускна здатність GPU на батчі 64 і повний шлях «файл JPEG → топ-3» на CPU.

Запуск: python scripts/bench_inference.py
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, eval_transform, load_meta  # noqa: E402
from agroscan.model import load_checkpoint  # noqa: E402


def timeit(fn, warmup, runs, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(ts), "p95_ms": float(np.percentile(ts, 95)),
            "min_ms": min(ts), "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO / "models" / "agroscan_efficientnet_b0.pt")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    res = {"cpu_name": platform.processor(), "torch_threads": torch.get_num_threads(),
           "gpu_name": torch.cuda.get_device_name(), "torch": torch.__version__}

    model_cpu, ckpt = load_checkpoint(args.ckpt, "cpu")
    x1 = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        res["cpu_fp32_bs1"] = timeit(lambda: model_cpu(x1), args.warmup, args.runs, False)

        # повний шлях на CPU: відкрити JPEG → трансформації → модель → softmax → топ-3
        _, root = load_meta()
        img_path = next((root / "Tomato___Late_blight").iterdir())
        tf = eval_transform()

        def full():
            x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
            return model_cpu(x).softmax(1).topk(3)
        res["cpu_end_to_end_jpeg_bs1"] = timeit(full, 5, 50, False)

        model_gpu, _ = load_checkpoint(args.ckpt, "cuda")
        xg = x1.cuda()
        res["gpu_fp32_bs1"] = timeit(lambda: model_gpu(xg), args.warmup, args.runs, True)

        def fp16():
            with torch.autocast("cuda", dtype=torch.float16):
                return model_gpu(xg)
        res["gpu_amp_fp16_bs1"] = timeit(fp16, args.warmup, args.runs, True)

        xb = torch.randn(64, 3, 224, 224, device="cuda")
        for name, amp in [("gpu_fp32_bs64", False), ("gpu_amp_fp16_bs64", True)]:
            def batch(amp=amp):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    return model_gpu(xb)
            r = timeit(batch, 5, 30, True)
            r["img_per_s"] = 64 / (r["median_ms"] / 1000)
            res[name] = r

    (REPO / "results" / "inference_benchmark.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                               encoding="utf-8")
    print(f"CPU: {res['cpu_name']} ({res['torch_threads']} потоків torch) | GPU: {res['gpu_name']}")
    for k, v in res.items():
        if isinstance(v, dict):
            extra = f" | {v['img_per_s']:.0f} img/s" if "img_per_s" in v else ""
            print(f"  {k:26s}: медіана {v['median_ms']:7.2f} мс | p95 {v['p95_ms']:7.2f} мс{extra}")


if __name__ == "__main__":
    main()
