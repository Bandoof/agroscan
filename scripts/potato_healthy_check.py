"""Чому «Potato: healthy» лишається в топ-3 у v2 і що з цим робити без перенавчання.

Для v2 порівнюємо два режими передбачення на PlantVillage test і PlantDoc test:
  raw      — звичайні логіти (збалансований по класах прогноз; так навчено logit adjustment);
  +prior   — логіти + log(train_prior): прогноз під розподіл класів навчального потоку.
І дивимося, з якою ЙМОВІРНІСТЮ Potato healthy потрапляє в топ-3 (чи це «шум» на 1–2 %).
Результат → results/v2/potato_healthy_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, PlantVillageSubset, eval_transform  # noqa: E402
from agroscan.data_v2 import PlantDocSubset  # noqa: E402
from agroscan.model import load_checkpoint  # noqa: E402


@torch.no_grad()
def logits_for(model, ds, device):
    out, ys = [], []
    for x, y in DataLoader(ds, batch_size=64, num_workers=6):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys).numpy()


def stats(logits, y, ph, labels):
    P = logits.softmax(1).numpy()
    top3 = np.argsort(-P, 1)[:, :3]
    pred = top3[:, 0]
    mask = y != ph
    in3 = np.array([ph in r for r in top3[mask]])
    p_ph = P[mask][in3, ph] if in3.any() else np.array([0.0])
    return {"top1": float((pred == y).mean()), "top3": float(np.mean([t in r for t, r in zip(y, top3)])),
            "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "ph_in_top3_when_not_true": float(in3.mean()),
            "ph_in_top3_with_p_ge_5pct": float(np.mean(P[mask][:, ph][in3.nonzero()[0]] >= 0.05) * in3.mean()) if in3.any() else 0.0,
            "ph_prob_when_in_top3_median": float(np.median(p_ph)),
            "ph_prob_when_in_top3_p90": float(np.percentile(p_ph, 90))}


def main():
    device = torch.device("cuda")
    model, ckpt = load_checkpoint(REPO / "models" / "agroscan_v2.pt", device)
    classes = ckpt["classes"]; ph = classes.index("Potato___healthy")
    log_prior = torch.log(torch.tensor(ckpt["train_prior"], dtype=torch.float32))
    res = {}
    # Рішення про режим приймаємо на VAL; test — лише підтвердження (порядок у звіті такий самий)
    for name, ds in (("plantvillage_val", PlantVillageSubset("val", eval_transform())),
                     ("plantdoc_val", PlantDocSubset("pd_val", eval_transform())),
                     ("plantvillage_test", PlantVillageSubset("test", eval_transform())),
                     ("plantdoc_test", PlantDocSubset("pd_test", eval_transform()))):
        L, y = logits_for(model, ds, device)
        labels = sorted(set(y)) if name.startswith("plantdoc") else list(range(len(classes)))
        res[name] = {"raw": stats(L, y, ph, labels), "plus_log_prior": stats(L + log_prior, y, ph, labels)}
    (REPO / "results" / "v2" / "potato_healthy_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for name, d in res.items():
        print(name)
        for mode, s in d.items():
            print(f"  {mode:15s} top1 {s['top1']:.3f} top3 {s['top3']:.3f} F1 {s['f1_macro']:.3f} | PH у топ-3 "
                  f"{s['ph_in_top3_when_not_true']:.1%} (з p>=5%: {s['ph_in_top3_with_p_ge_5pct']:.1%}), "
                  f"медіана p(PH) коли в топ-3 {s['ph_prob_when_in_top3_median']:.3f}, p90 {s['ph_prob_when_in_top3_p90']:.3f}")


if __name__ == "__main__":
    main()
