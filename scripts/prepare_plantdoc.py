"""Крок 4б-1: поділ PlantDoc (14 спільних класів) для навчання v2 — без витоку в test.

- pd_test  — офіційна тека test PlantDoc. НЕ використовується ні для навчання, ні для вибору епохи.
- pd_train / pd_val — офіційна тека train, поділена 80/20 (seed 42) по класах.
- Дублікати: PlantDoc зібрано з інтернету, одне фото може лежати кілька разів.
  Шукаємо точні (MD5) і майже-точні (dHash 64 біт, відстань Геммінга <= HAMMING) копії.
  * Фото з train, яке дублює будь-яке фото з test, ВИКЛЮЧАЄМО з навчання (pd_excluded_dup).
  * Дублікати всередині train об'єднуємо в групи і ділимо train/val цілими групами.
Результат: data/plantdoc_splits.csv, results/v2/plantdoc_split_summary.json

Запуск: python scripts/prepare_plantdoc.py --images C:/Users/kuril/datasets/plantdoc/images
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agroscan.data import REPO  # noqa: E402
from eval_plantdoc import MAPPING  # noqa: E402

HAMMING = 4
VAL_FRACTION = 0.2


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = []  # (rel_path, pd_split, pd_class)
    for split in ("train", "test"):
        for cls_dir in sorted((args.images / split).iterdir()):
            if not MAPPING.get(cls_dir.name):
                continue
            for f in sorted(cls_dir.iterdir()):
                items.append((f"{split}/{cls_dir.name}/{f.name}", split, cls_dir.name))

    md5s, dh = [], []
    for rel, _, _ in items:
        b = (args.images / rel).read_bytes()
        md5s.append(hashlib.md5(b).hexdigest())
        with Image.open(args.images / rel) as im:
            dh.append(dhash(im))
    dh_arr = np.array(dh, dtype=np.uint64)

    # пари дублікатів (усі проти всіх; n≈1.2k — це ~0.7 млн порівнянь)
    n = len(items)
    dsu = DSU(n)
    pairs_exact, pairs_near = [], []
    for i in range(n):
        x = np.bitwise_xor(dh_arr[i + 1:], dh_arr[i])
        dist = np.array([bin(int(v)).count("1") for v in x])
        for k in np.nonzero(dist <= HAMMING)[0]:
            j = i + 1 + int(k)
            (pairs_exact if md5s[i] == md5s[j] else pairs_near).append((i, j, int(dist[k])))
            dsu.union(i, j)
        for j in range(i + 1, n):
            if md5s[i] == md5s[j] and dsu.find(i) != dsu.find(j):
                pairs_exact.append((i, j, 0)); dsu.union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    test_groups = {g for g, mem in groups.items() if any(items[m][1] == "test" for m in mem)}

    split_of = {}
    for i, (_, s, _) in enumerate(items):
        if s == "test":
            split_of[i] = "pd_test"
        elif dsu.find(i) in test_groups:
            split_of[i] = "pd_excluded_dup"  # дублює фото з test → не вчимо на ньому

    # train/val по групах, стратифіковано за класом (клас групи = клас першого члена)
    rng = np.random.default_rng(args.seed)
    by_class = defaultdict(list)
    for g, mem in groups.items():
        free = [m for m in mem if m not in split_of]
        if free:
            by_class[items[free[0]][2]].append(free)
    for cls in sorted(by_class):
        grps = sorted(by_class[cls], key=lambda m: items[m[0]][0])
        rng.shuffle(grps)
        total = sum(len(g) for g in grps)
        n_val = int(round(total * VAL_FRACTION)) if total >= 5 else 0  # 2 фото павутинного кліща — лише train
        acc = 0
        for g in grps:
            target = "pd_val" if acc < n_val else "pd_train"
            for m in g:
                split_of[m] = target
            if target == "pd_val":
                acc += len(g)

    cross = [(i, j, d) for i, j, d in pairs_exact + pairs_near if {items[i][1], items[j][1]} == {"train", "test"}]
    rows = [{"path": rel, "pd_class": c, "label": MAPPING[c], "split": split_of[i], "group": dsu.find(i)}
            for i, (rel, _, c) in enumerate(items)]
    with open(REPO / "data" / "plantdoc_splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (REPO / "data" / "plantdoc_meta.json").write_text(
        json.dumps({"images_root": str(args.images), "seed": args.seed, "val_fraction": VAL_FRACTION,
                    "hamming_threshold": HAMMING}, ensure_ascii=False, indent=2), encoding="utf-8")

    cnt = Counter(r["split"] for r in rows)
    per_class = defaultdict(Counter)
    for r in rows:
        per_class[r["label"]][r["split"]] += 1
    summary = {"hamming_threshold": HAMMING, "n_images": n, "splits": dict(cnt),
               "duplicate_pairs_exact": len(pairs_exact), "duplicate_pairs_near": len(pairs_near),
               "train_test_duplicate_pairs": len(cross),
               "test_images_with_train_duplicate": len({j if items[j][1] == "test" else i for i, j, _ in cross}),
               "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
               "cross_examples": [(items[i][0], items[j][0], d) for i, j, d in cross[:10]]}
    out = REPO / "results" / "v2"; out.mkdir(parents=True, exist_ok=True)
    (out / "plantdoc_split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PlantDoc, 14 спільних класів: {n} фото")
    print(f"Пари дублікатів: точних {len(pairs_exact)}, майже-точних (dHash <= {HAMMING}) {len(pairs_near)}")
    print(f"  з них між train і test: {len(cross)} пар; фото test, що мають копію в train: "
          f"{summary['test_images_with_train_duplicate']}")
    for s in ("pd_train", "pd_val", "pd_test", "pd_excluded_dup"):
        print(f"  {s:16s}: {cnt[s]}")
    print(f"\n  {'клас':48s} {'train':>5s} {'val':>5s} {'test':>5s} {'excl':>5s}")
    for k, v in sorted(per_class.items()):
        print(f"  {k:48s} {v['pd_train']:5d} {v['pd_val']:5d} {v['pd_test']:5d} {v['pd_excluded_dup']:5d}")


if __name__ == "__main__":
    main()
