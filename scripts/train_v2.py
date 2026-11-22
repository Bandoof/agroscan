"""Крок 4б: навчання моделі v2 (PlantVillage + PlantDoc train, випадкові фони, logit adjustment).

Відмінності від v1 (scripts/train.py):
- дані: PlantVillage train (+ випадкові фони на сегментованих фото, p=0.5) + PlantDoc pd_train;
- WeightedRandomSampler: ~25 % PlantDoc у кожній епосі, класи ∝ 1/sqrt(n) усередині джерела;
- loss: logit-adjusted CE (Menon et al., 2021) — до логітів додається tau·log(prior) лише під час
  навчання; без ваг класів. Під час передбачення — звичайні логіти;
- вибір епохи й early stopping: macro-F1 на PlantDoc pd_val (PlantVillage val — для контролю).
PlantDoc pd_test тут НЕ завантажується взагалі.

Запуск: python scripts/train_v2.py           (повне навчання → models/agroscan_v2.pt)
        python scripts/train_v2.py --smoke   (по 20 батчів — перевірка)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agroscan.data import (IMAGENET_MEAN, IMAGENET_STD, REPO, PlantVillageSubset,  # noqa: E402
                           train_transform)
from agroscan.data_v2 import PlantDocSubset, PlantVillageBG, make_sampler  # noqa: E402
from agroscan.model import build_model, set_backbone_trainable  # noqa: E402
from train import seed_everything  # noqa: E402


class LogitAdjustedCE(nn.Module):
    def __init__(self, prior, tau=1.0, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("adj", tau * torch.log(torch.as_tensor(prior, dtype=torch.float32)))
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, y):
        return self.ce(logits + self.adj, y)


def cache_eval_set(ds, workers):
    """Декодує val-вибірку ОДИН раз у uint8-тензор у RAM (Resize 256 → CenterCrop 224, як eval_transform).
    Так валідація щоепохи не потребує окремих процесів-воркерів: на Windows кожен воркер
    вантажить torch+CUDA (~1.4 ГБ commit), і 18 воркерів у першій спробі вичерпали файл підкачки."""
    from torchvision import transforms as T
    ds.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers)  # тимчасові воркери, лише на старті
    xs, ys = zip(*dl)
    return torch.utils.data.TensorDataset(torch.cat(xs), torch.cat(ys))


MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


@torch.no_grad()
def evaluate(model, loader, device, labels=None, max_batches=None):
    model.eval()
    mean, std = MEAN.to(device), STD.to(device)
    preds, ys = [], []
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x = (x.to(device, non_blocking=True).float() / 255 - mean) / std  # нормалізація на GPU
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        preds.append(out.argmax(1).cpu()); ys.append(y)
    p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    labels = labels if labels is not None else sorted(set(t))
    return {"acc": float((p == t).mean()),
            "f1_macro": float(f1_score(t, p, labels=labels, average="macro", zero_division=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-frozen", type=int, default=2)
    ap.add_argument("--epochs-finetune", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--pd-share", type=float, default=0.25)
    ap.add_argument("--p-bg", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs_frozen, args.epochs_finetune = 1, 1
    max_batches = 20 if args.smoke else None

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    pv_train = PlantVillageBG(train_transform(), p_bg=args.p_bg)
    pd_train = PlantDocSubset("pd_train", train_transform())
    classes = pv_train.classes
    pv_val_raw, pd_val_raw = PlantVillageSubset("val"), PlantDocSubset("pd_val")
    pd_val_labels = sorted(set(pd_val_raw.targets()))
    t0 = time.perf_counter()
    pv_val = cache_eval_set(pv_val_raw, args.workers)
    pd_val = cache_eval_set(pd_val_raw, args.workers)
    print(f"val закешовано в RAM за {time.perf_counter()-t0:.0f} с: PV {tuple(pv_val.tensors[0].shape)}, "
          f"PD {tuple(pd_val.tensors[0].shape)}, "
          f"{(pv_val.tensors[0].nbytes + pd_val.tensors[0].nbytes)/2**20:.0f} МБ", flush=True)

    sampler, prior = make_sampler(pv_train.targets(), pd_train.targets(), len(classes), args.pd_share,
                                  seed=args.seed)
    train_ds = ConcatDataset([pv_train, pd_train])
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True, **kw)
    pv_val_dl = DataLoader(pv_val, batch_size=128, shuffle=False, num_workers=0)  # дані вже в RAM
    pd_val_dl = DataLoader(pd_val, batch_size=128, shuffle=False, num_workers=0)

    criterion = LogitAdjustedCE(prior, tau=args.tau).to(device)
    model = build_model("efficientnet_b0", len(classes)).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = REPO / "results" / "v2"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "v2_smoke" if args.smoke else "v2"
    ckpt_path = REPO / "models" / f"agroscan_{tag}.pt"
    log_f = open(out_dir / f"train_log_{tag}.csv", "w", newline="", encoding="utf-8")
    fields = ["epoch", "stage", "train_loss", "train_acc", "pv_val_acc", "pv_val_f1_macro", "pd_val_acc",
              "pd_val_f1_macro", "train_time_s", "img_per_s", "peak_vram_mb"]
    log = csv.DictWriter(log_f, fieldnames=fields); log.writeheader()

    print(f"v2: train PV {len(pv_train)} (+фони p={args.p_bg}) + PD {len(pd_train)}; семплів за епоху {len(sampler)}, "
          f"частка PD {args.pd_share} | val PV {len(pv_val)}, PD {len(pd_val)} | logit adj tau={args.tau}")
    print("апріорний розподіл класів у потоці (для logit adjustment):",
          " ".join(f"{p:.3f}" for p in prior), flush=True)

    best, best_epoch, epoch = -1.0, 0, 0
    t_start = time.perf_counter()
    for stage, n_epochs in (("frozen", args.epochs_frozen), ("finetune", args.epochs_finetune)):
        frozen = stage == "frozen"
        set_backbone_trainable(model, not frozen)
        lr = args.lr_head if frozen else args.lr_finetune
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        steps = n_epochs * min(len(train_dl), max_batches or 10**9)
        sched = None if frozen else torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                                        pct_start=0.1, anneal_strategy="cos")
        bad = 0
        for _ in range(n_epochs):
            epoch += 1
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model.train()
            if frozen:
                model.features.eval()
            t0, tot_loss, correct, seen = time.perf_counter(), 0.0, 0, 0
            for b, (x, y) in enumerate(train_dl):
                if max_batches and b >= max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(x)
                    loss = criterion(out.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                if sched:
                    sched.step()
                tot_loss += loss.item() * x.size(0); seen += x.size(0)
                correct += (out.argmax(1) == y).sum().item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            pv = evaluate(model, pv_val_dl, device, max_batches=max_batches)
            pd = evaluate(model, pd_val_dl, device, labels=pd_val_labels, max_batches=max_batches)
            vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
            row = {"epoch": epoch, "stage": stage, "train_loss": round(tot_loss / seen, 4),
                   "train_acc": round(correct / seen, 4), "pv_val_acc": round(pv["acc"], 4),
                   "pv_val_f1_macro": round(pv["f1_macro"], 4), "pd_val_acc": round(pd["acc"], 4),
                   "pd_val_f1_macro": round(pd["f1_macro"], 4), "train_time_s": round(dt, 1),
                   "img_per_s": round(seen / dt, 1), "peak_vram_mb": round(vram)}
            log.writerow(row); log_f.flush()
            improved = pd["f1_macro"] > best
            print(f"[{epoch:2d} {stage:8s}] loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
                  f"PV val acc {pv['acc']:.4f} F1 {pv['f1_macro']:.4f} | PD val acc {pd['acc']:.4f} "
                  f"F1 {pd['f1_macro']:.4f} | {dt:.0f}s {seen/dt:.0f} img/s VRAM {vram:.0f} MB"
                  + ("  ← best" if improved else ""), flush=True)
            if improved:
                best, best_epoch, bad = pd["f1_macro"], epoch, 0
                torch.save({"model": model.state_dict(), "arch": "efficientnet_b0", "classes": classes,
                            "epoch": epoch, "version": "v2", "pd_val_f1_macro": pd["f1_macro"],
                            "pv_val_f1_macro": pv["f1_macro"], "val_acc": pv["acc"], "val_f1_macro": pv["f1_macro"],
                            "train_prior": prior.tolist(), "img_size": 224, "args": vars(args)}, ckpt_path)
            else:
                bad += 1
                if not frozen and bad >= args.patience:
                    print(f"Рання зупинка: {args.patience} епохи без покращення PD val macro-F1.")
                    break
    log_f.close()
    total = time.perf_counter() - t_start
    (out_dir / f"train_summary_{tag}.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_pd_val_f1_macro": best, "epochs_run": epoch,
        "total_train_time_min": round(total / 60, 2), "n_pv_train": len(pv_train), "n_pd_train": len(pd_train),
        "samples_per_epoch": len(sampler), "train_prior": prior.tolist(), "args": vars(args)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Готово за {total/60:.1f} хв. Найкраща епоха {best_epoch}, PD val macro-F1 {best:.4f} → {ckpt_path}")


if __name__ == "__main__":
    main()
