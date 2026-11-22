"""Незалежна перевірка витоку: чи є серед навчальних даних v2 (PlantDoc pd_train/pd_val + PlantVillage train)
дублікати або майже дублікати фото з PlantDoc test.

Два незалежні методи (на додачу до MD5 + dHash<=4 у prepare_plantdoc.py):
1) ембединги EfficientNet-B0, натренованої лише на ImageNet (НЕ наша модель) → косинусна схожість;
   ембединг рахується і для дзеркального фото, береться максимум;
2) dHash з ширшим порогом (<= 12) + dHash дзеркального фото.
Для кожного фото test шукаємо найближче навчальне фото. Кандидати (cos >= COS_THR або dHash <= 12)
рендеряться парами для ручної перевірки → results/v2/leak_candidates.png, results/v2/leak_check.json.
Еталон: 11 пар «test ↔ pd_excluded_dup», уже знайдених раніше, — їхню схожість видно в звіті.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agroscan.data import REPO, eval_transform, load_meta  # noqa: E402
from agroscan.data_v2 import load_plantdoc_root  # noqa: E402
from prepare_plantdoc import dhash  # noqa: E402

# 0.78: з порогом 0.90 знайшовся кроп-дублікат лише з cos 0.809 (через dHash), тож поріг знижено
# з запасом; медіана max cos «звичайного» фото test до навчальних даних — 0.72.
COS_THR = 0.78
DHASH_THR = 12


class Paths(Dataset):
    def __init__(self, paths):
        self.paths, self.tf = paths, eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.tf(ImageOps.mirror(img))


@torch.no_grad()
def embed(paths, net, device):
    out, out_m = [], []
    for x, xm in DataLoader(Paths(paths), batch_size=64, num_workers=6):
        for t, acc in ((x, out), (xm, out_m)):
            with torch.autocast("cuda", dtype=torch.float16):
                e = net(t.to(device)).float()
            acc.append(torch.nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(out), torch.cat(out_m)


def dhashes(paths):
    h, hm = [], []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            h.append(dhash(im)); hm.append(dhash(ImageOps.mirror(im)))
    return np.array(h, dtype=np.uint64), np.array(hm, dtype=np.uint64)


def ham(a, B):
    return np.array([bin(int(v)).count("1") for v in np.bitwise_xor(B, a)])


def main():
    device = torch.device("cuda")
    pd_root = load_plantdoc_root()
    _, pv_root = load_meta()
    with open(REPO / "data" / "plantdoc_splits.csv", encoding="utf-8") as f:
        pd_rows = list(csv.DictReader(f))
    with open(REPO / "data" / "splits.csv", encoding="utf-8") as f:
        pv_rows = [r for r in csv.DictReader(f) if r["split"] == "train"]

    test = [r for r in pd_rows if r["split"] == "pd_test"]
    pools = {"pd_train": [r for r in pd_rows if r["split"] == "pd_train"],
             "pd_val": [r for r in pd_rows if r["split"] == "pd_val"],
             "pd_excluded_dup": [r for r in pd_rows if r["split"] == "pd_excluded_dup"],
             "pv_train": pv_rows}
    path_of = lambda name, r: (pv_root if name == "pv_train" else pd_root) / r["path"]  # noqa: E731

    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()
    net = net.to(device).eval()

    test_paths = [pd_root / r["path"] for r in test]
    Et, Etm = embed(test_paths, net, device)
    Ht, Htm = dhashes(test_paths)

    best = [{"cos": -1.0} for _ in test]
    ref = []  # схожість уже відомих дублікатів (еталон)
    for name, rows in pools.items():
        paths = [path_of(name, r) for r in rows]
        E, _ = embed(paths, net, device)
        S = torch.maximum(Et @ E.T, Etm @ E.T).numpy()  # враховуємо дзеркальне фото test
        H = dhashes(paths)[0] if name != "pv_train" else None  # dHash для PD (PV — за ембедингами)
        for i in range(len(test)):
            j = int(S[i].argmax())
            dh = None
            if H is not None:
                dh = int(min(ham(Ht[i], H).min(), ham(Htm[i], H).min()))
            if S[i, j] > best[i]["cos"]:
                best[i] = {"cos": float(S[i, j]), "pool": name, "match": rows[j]["path"],
                           "match_label": rows[j].get("label", ""), "dhash_min_in_pool": dh}
            if name == "pd_excluded_dup":
                ref.append(float(S[i].max()))
        print(f"  {name}: {len(rows)} фото оброблено", flush=True)

    known = {r["path"] for r in test if any(r["group"] == e["group"] for e in pools["pd_excluded_dup"])}
    cands = [(i, b) for i, b in enumerate(best)
             if b["pool"] != "pd_excluded_dup" and (b["cos"] >= COS_THR or
                                                     (b["dhash_min_in_pool"] is not None and b["dhash_min_in_pool"] <= DHASH_THR))]
    cands.sort(key=lambda t: -t[1]["cos"])
    ref_cos = sorted(best[i]["cos"] for i, r in enumerate(test) if r["path"] in known)
    cos_all = np.array([b["cos"] for b in best if b["pool"] != "pd_excluded_dup"])

    report = {"cos_threshold": COS_THR, "dhash_threshold": DHASH_THR, "n_test": len(test),
              "known_excluded_dup_cos": ref_cos,
              "max_cos_to_training_pool_percentiles": {p: float(np.percentile(cos_all, p)) for p in (50, 90, 99, 100)},
              "candidates": [{"test": test[i]["path"], "test_label": test[i]["label"], **b} for i, b in cands]}
    (REPO / "results" / "v2" / "leak_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    print(f"Еталон (відомі дублікати, cos до виключених копій): {[round(c, 3) for c in ref_cos]}")
    print("Макс. cos кожного фото test до навчальних даних v2: перцентилі",
          {k: round(v, 3) for k, v in report["max_cos_to_training_pool_percentiles"].items()})
    print(f"Кандидатів (cos >= {COS_THR} або dHash <= {DHASH_THR}): {len(cands)}")
    for i, b in cands:
        print(f"  cos {b['cos']:.3f} dHash {b['dhash_min_in_pool']} | {b['pool']:8s} | test {test[i]['label']} ↔ "
              f"{b['match_label'] or b['match'].split('/')[0]}")

    if cands:
        fig, axes = plt.subplots(len(cands), 2, figsize=(6, 2.6 * len(cands)))
        axes = np.atleast_2d(axes)
        for (i, b), (a1, a2) in zip(cands, axes):
            a1.imshow(Image.open(test_paths[i]).convert("RGB")); a1.axis("off")
            a1.set_title(f"#{i} test: {test[i]['label'].split('___')[-1]}", fontsize=7)
            a2.imshow(Image.open(path_of(b["pool"], {"path": b["match"]})).convert("RGB")); a2.axis("off")
            a2.set_title(f"{b['pool']} cos {b['cos']:.3f} dH {b['dhash_min_in_pool']}", fontsize=7)
        fig.tight_layout(); fig.savefig(REPO / "results" / "v2" / "leak_candidates.png", dpi=70)


if __name__ == "__main__":
    main()
