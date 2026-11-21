"""Крок 3: навчання AgroScan (transfer learning, 2 етапи, AMP).

Етап 1 (frozen): основа ImageNet заморожена, вчиться лише нова голова на 17 класів.
Етап 2 (fine-tune): розморожуємо всю мережу, менший LR + косинусний розклад.
Найкраща модель за val macro-F1 → models/agroscan_<arch>.pt, лог → results/.

Запуск:  python scripts/train.py --arch efficientnet_b0
Швидка перевірка: python scripts/train.py --smoke
"""
import argparse
import csv
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, PlantVillageSubset, eval_transform, train_transform  # noqa: E402
from agroscan.model import ARCHS, build_model, set_backbone_trainable  # noqa: E402


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, scheduler=None,
              max_batches=None, freeze_bn_backbone=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_bn_backbone:
        model.features.eval()  # заморожена основа: не оновлюємо статистики BatchNorm
    total_loss, preds, targets = 0.0, [], []
    t0 = time.perf_counter()
    n_seen = 0
    with torch.set_grad_enabled(train):
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            # channels_last НЕ використовуємо: на RTX 2050 з EfficientNet-B0 він дав 16 img/s проти 248
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            preds.append(out.argmax(1).cpu()); targets.append(y.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {"loss": total_loss / n_seen, "acc": float((p == t).mean()),
            "f1_macro": float(f1_score(t, p, average="macro")), "time_s": dt, "img_per_s": n_seen / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    ap.add_argument("--epochs-frozen", type=int, default=3)
    ap.add_argument("--epochs-finetune", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="по 20 батчів, 1+1 епоха — перевірка VRAM/швидкості")
    args = ap.parse_args()
    if args.smoke:
        args.epochs_frozen, args.epochs_finetune = 1, 1
    max_batches = 20 if args.smoke else None

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_ds = PlantVillageSubset("train", train_transform())
    val_ds = PlantVillageSubset("val", eval_transform())
    classes = train_ds.classes
    dl_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                 persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    # Ваги класів ("balanced"): рідкісні класи (Potato healthy, 108 фото) важать більше
    cnt = Counter(train_ds.targets())
    weights = torch.tensor([len(train_ds) / (len(classes) * cnt[i]) for i in range(len(classes))],
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    model = build_model(args.arch, len(classes)).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    tag = args.arch + ("_smoke" if args.smoke else "")
    (REPO / "models").mkdir(exist_ok=True)
    (REPO / "results").mkdir(exist_ok=True)
    ckpt_path = REPO / "models" / f"agroscan_{tag}.pt"
    log_path = REPO / "results" / f"train_log_{tag}.csv"
    fields = ["epoch", "stage", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro",
              "train_time_s", "train_img_per_s", "peak_vram_mb"]
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.DictWriter(log_f, fieldnames=fields); log.writeheader()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_params/1e6:.2f} M параметрів | train {len(train_ds)}, val {len(val_ds)} | "
          f"batch {args.batch_size} | AMP fp16 | {device} "
          f"({torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor()})")

    best_f1, best_epoch, bad_epochs, epoch = -1.0, 0, 0, 0
    t_start = time.perf_counter()
    stages = [("frozen", args.epochs_frozen), ("finetune", args.epochs_finetune)]
    for stage, n_epochs in stages:
        if n_epochs == 0:
            continue
        frozen = stage == "frozen"
        set_backbone_trainable(model, not frozen)
        params = [p for p in model.parameters() if p.requires_grad]
        lr = args.lr_head if frozen else args.lr_finetune
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        steps = n_epochs * (min(len(train_dl), max_batches or 10**9))
        scheduler = None if frozen else torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="cos")
        bad_epochs = 0
        for _ in range(n_epochs):
            epoch += 1
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            tr = run_epoch(model, train_dl, criterion, device, optimizer, scaler, scheduler,
                           max_batches, freeze_bn_backbone=frozen)
            va = run_epoch(model, val_dl, criterion, device, max_batches=max_batches)
            vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
            row = {"epoch": epoch, "stage": stage, "lr": lr, "train_loss": round(tr["loss"], 4),
                   "train_acc": round(tr["acc"], 4), "val_loss": round(va["loss"], 4),
                   "val_acc": round(va["acc"], 4), "val_f1_macro": round(va["f1_macro"], 4),
                   "train_time_s": round(tr["time_s"], 1), "train_img_per_s": round(tr["img_per_s"], 1),
                   "peak_vram_mb": round(vram)}
            log.writerow(row); log_f.flush()
            improved = va["f1_macro"] > best_f1
            print(f"[{epoch:2d} {stage:8s}] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.4f} F1 {va['f1_macro']:.4f} | "
                  f"{tr['time_s']:.0f}s {tr['img_per_s']:.0f} img/s VRAM {vram:.0f} MB"
                  + ("  ← best" if improved else ""), flush=True)
            if improved:
                best_f1, best_epoch, bad_epochs = va["f1_macro"], epoch, 0
                torch.save({"model": model.state_dict(), "arch": args.arch, "classes": classes,
                            "epoch": epoch, "val_acc": va["acc"], "val_f1_macro": va["f1_macro"],
                            "img_size": 224, "args": vars(args)}, ckpt_path)
            else:
                bad_epochs += 1
                if not frozen and bad_epochs >= args.patience:
                    print(f"Рання зупинка: {args.patience} епохи без покращення val F1.")
                    break
    log_f.close()

    total = time.perf_counter() - t_start
    info = {"arch": args.arch, "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
            "epochs_run": epoch, "total_train_time_min": round(total / 60, 2), "params_M": round(n_params / 1e6, 2),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "torch": torch.__version__, "args": vars(args)}
    (REPO / "results" / f"train_summary_{tag}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                                                 encoding="utf-8")
    print(f"Готово за {total/60:.1f} хв. Найкраща епоха {best_epoch}, val macro-F1 {best_f1:.4f} → {ckpt_path}")


if __name__ == "__main__":  # обов'язково на Windows для DataLoader з num_workers>0
    main()
