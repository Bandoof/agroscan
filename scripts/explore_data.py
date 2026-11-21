"""Крок 2b: графік розподілу класів і сітка прикладів → results/."""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
from agroscan.data import load_meta  # noqa: E402

meta = json.loads((REPO / "data" / "classes.json").read_text(encoding="utf-8"))
classes, images_root = load_meta()
with open(REPO / "data" / "splits.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

SPLITS = ["train", "val", "test"]
COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

counts = Counter((r["label"], r["split"]) for r in rows)
pretty = {c: c.replace("___", " — ").replace("_", " ") for c in classes}

# --- таблиця (для звіту й доступності) ---
with open(RESULTS / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["class", *SPLITS, "total"])
    for c in classes:
        vals = [counts[(c, s)] for s in SPLITS]
        w.writerow([c, *vals, sum(vals)])

# --- 1. розподіл класів: горизонтальні складені смуги ---
order = classes[::-1]  # Apple зверху
fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
left = [0] * len(order)
for s in SPLITS:
    vals = [counts[(c, s)] for c in order]
    ax.barh([pretty[c] for c in order], vals, left=left, height=0.62, color=COLORS[s],
            edgecolor=SURFACE, linewidth=1.5, label=s)
    left = [l + v for l, v in zip(left, vals)]
for y, tot in enumerate(left):
    ax.text(tot + 40, y, f"{tot}", va="center", fontsize=9, color=INK2)
ax.set_xlabel("Кількість зображень", color=INK2)
ax.set_title(f"PlantVillage (томат / картопля / яблуня): {len(classes)} класів, {len(rows)} зображень\n"
             f"поділ 70/15/15 за листками, seed {meta['seed']}", color=INK, loc="left", fontsize=12)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=INK2, labelsize=9)
ax.set_xlim(0, max(left) * 1.08)
ax.legend(title="Вибірка", frameon=False, loc="lower right", labelcolor=INK2)
fig.tight_layout()
fig.savefig(RESULTS / "class_distribution.png", dpi=150)
plt.close(fig)

# --- 2. сітка прикладів: по 2 фото на клас (з train), фіксований seed ---
rng = random.Random(42)
per_class, ncols = 2, 6
samples = []
for c in classes:
    pool = [r["path"] for r in rows if r["label"] == c and r["split"] == "train"]
    samples += [(c, p) for p in rng.sample(pool, per_class)]
nrows = -(-len(samples) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.3, nrows * 2.55), facecolor=SURFACE)
for ax in axes.flat:
    ax.axis("off")
for ax, (c, p) in zip(axes.flat, samples):
    ax.imshow(Image.open(images_root / p).convert("RGB"))
    crop, disease = c.split("___")
    disease = disease.replace("_", " ").replace("Tomato ", "").replace(" Two-spotted spider mite", "")
    ax.set_title(f"{crop}\n{disease}", fontsize=8, color=INK)
fig.suptitle("Приклади зображень за класами (по 2 на клас, train)", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(RESULTS / "sample_grid.png", dpi=110)
plt.close(fig)

print("Збережено: results/class_distribution.png, results/class_distribution.csv, "
      "results/sample_grid.png")
