"""Крок 4a: оцінка на тестовій вибірці PlantVillage + криві навчання.

Виводить і зберігає в results/:
- test_metrics.json           — accuracy, macro/weighted P/R/F1, незважений test loss;
- test_classification_report.csv — P/R/F1/support по кожному класу;
- confusion_matrix_test.png   — нормована матриця помилок 17×17;
- training_curves.png         — loss і accuracy по епохах (з train_log_*.csv);
- val_loss_check.json         — перевірка, чому val loss > train loss (див. нижче).

Запуск: python scripts/evaluate.py --ckpt models/agroscan_efficientnet_b0.pt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, PlantVillageSubset, eval_transform, short_name  # noqa: E402
from agroscan.model import load_checkpoint  # noqa: E402
from agroscan.plots import INK, INK2, SERIES, SURFACE, confusion_plot, style_axes  # noqa: E402

RESULTS = REPO / "results"


@torch.no_grad()
def predict(model, ds, device, bs=128, workers=6):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    logits, ys = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu()); ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def batched_loss(criterion, logits, y, bs, shuffle, seed=0):
    """Середнє по батчах — саме так рахує train.py (loss.item() * batch_size / N)."""
    idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(len(y))
    total = 0.0
    for i in range(0, len(y), bs):
        b = idx[i:i + bs]
        total += criterion(logits[b], y[b]).item() * len(b)
    return total / len(y)


def plot_curves(log_path, out_path):
    with open(log_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    n_frozen = sum(r["stage"] == "frozen" for r in rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, key, title in [(axes[0], "loss", "Loss (зважений CE + label smoothing)"),
                           (axes[1], "acc", "Accuracy")]:
        style_axes(ax)
        for color, split in [(SERIES[0], "train"), (SERIES[1], "val")]:
            vals = [float(r[f"{split}_{key}"]) for r in rows]
            ax.plot(ep, vals, color=color, linewidth=2, marker="o", markersize=4, label=split)
            ax.annotate(f"{split} {vals[-1]:.3f}", (ep[-1], vals[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.axvspan(0.5, n_frozen + 0.5, color="#e1e0d9", alpha=0.5, lw=0)
        ax.text(1, ax.get_ylim()[1], "основа\nзаморожена", va="top", fontsize=8, color=INK2)
        ax.set_xlabel("Епоха", color=INK2)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.set_xticks(ep)
        ax.set_xlim(0.5, ep[-1] + 2.2)
        ax.legend(frameon=False, loc="upper right" if key == "loss" else "lower right", labelcolor=INK2)
    axes[0].text(0.99, 0.62, "val loss завищений артефактом підрахунку:\nбатчі val не перемішані, а зважений CE\n"
                 "нормується по батчу (див. val_loss_check.json)", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=7.5, color=INK2)
    best = max(rows, key=lambda r: float(r["val_f1_macro"]))
    axes[1].axvline(int(best["epoch"]), color="#898781", linestyle="--", linewidth=1)
    axes[1].text(int(best["epoch"]) + 0.1, 0.75, f"найкраща\n(епоха {best['epoch']})", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO / "models" / "agroscan_efficientnet_b0.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    classes = ckpt["classes"]
    names = [short_name(c) for c in classes]
    arch = ckpt["arch"]

    # --- тест ---
    test_ds = PlantVillageSubset("test", eval_transform())
    logits, y = predict(model, test_ds, device)
    pred = logits.argmax(1).numpy(); yt = y.numpy()
    acc = accuracy_score(yt, pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(yt, pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(yt, pred, average="weighted", zero_division=0)
    test_loss = nn.functional.cross_entropy(logits, y).item()
    metrics = {"checkpoint": str(args.ckpt.name), "arch": arch, "best_epoch": ckpt["epoch"], "n_test": len(yt),
               "n_errors": int((pred != yt).sum()), "accuracy": acc,
               "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
               "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
               "test_loss_unweighted_ce": test_loss}
    (RESULTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = classification_report(yt, pred, target_names=classes, output_dict=True, zero_division=0)
    with open(RESULTS / "test_classification_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for c in classes:
            w.writerow([c, f"{rep[c]['precision']:.4f}", f"{rep[c]['recall']:.4f}",
                        f"{rep[c]['f1-score']:.4f}", int(rep[c]["support"])])

    cm = confusion_matrix(yt, pred, labels=range(len(classes)))
    confusion_plot(cm, names, names,
                   f"Матриця помилок, PlantVillage test (n={len(yt)}, accuracy {acc:.4f})",
                   RESULTS / "confusion_matrix_test.png")

    # найчастіші помилки
    errs = [(cm[i, j], names[i], names[j]) for i in range(len(classes)) for j in range(len(classes))
            if i != j and cm[i, j] > 0]
    errs.sort(reverse=True)

    # --- перевірка: чому val loss у train.py > train loss ---
    val_ds = PlantVillageSubset("val", eval_transform())
    vlog, vy = predict(model, val_ds, device)
    cnt = np.bincount(PlantVillageSubset("train").targets(), minlength=len(classes))
    w = torch.tensor(cnt.sum() / (len(classes) * cnt), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    check = {
        "val_loss_as_in_train_py (weighted+LS, unshuffled batches of 64)": batched_loss(crit, vlog, vy, 64, False),
        "val_loss_weighted+LS, shuffled batches of 64": batched_loss(crit, vlog, vy, 64, True),
        "val_loss_weighted+LS, whole set at once": crit(vlog, vy).item(),
        "val_loss_unweighted_CE_with_LS0.1": nn.CrossEntropyLoss(label_smoothing=0.1)(vlog, vy).item(),
        "val_loss_unweighted_CE_plain": nn.functional.cross_entropy(vlog, vy).item(),
    }
    (RESULTS / "val_loss_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_curves(RESULTS / f"train_log_{arch}.csv", RESULTS / "training_curves.png")

    print(f"PlantVillage TEST (n={len(yt)}), чекпоінт епохи {ckpt['epoch']}:")
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "test_loss_unweighted_ce"):
        print(f"  {k:24s}: {metrics[k]:.4f}")
    print(f"  помилок: {metrics['n_errors']} з {len(yt)}")
    print("\nПо класах (recall найнижчі першими):")
    for c in sorted(classes, key=lambda c: rep[c]["recall"]):
        print(f"  {short_name(c):32s} P {rep[c]['precision']:.3f}  R {rep[c]['recall']:.3f}  "
              f"F1 {rep[c]['f1-score']:.3f}  n={int(rep[c]['support'])}")
    print("\nНайчастіші сплутування (справжній → передбачений):")
    for n, a, b in errs[:8]:
        print(f"  {int(n):3d}× {a} → {b}")
    print("\nПеревірка val loss:")
    for k, v in check.items():
        print(f"  {k:66s}: {v:.4f}")


if __name__ == "__main__":
    main()
