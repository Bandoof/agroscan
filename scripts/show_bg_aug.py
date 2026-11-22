"""Візуальна перевірка аугментації фонами: оригінал / сегментоване / 3 випадкові фони → results/v2/bg_augmentation.png"""
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, short_name  # noqa: E402
from agroscan.data_v2 import PlantVillageBG, composite_on_background  # noqa: E402
from agroscan.plots import INK, SURFACE  # noqa: E402

ds = PlantVillageBG(transform=None, p_bg=1.0)
rng = random.Random(7)
# по одному фото з 6 різних класів, включно з темними плямами (чорна гниль, фітофтороз)
want = ["Apple___Black_rot", "Apple___healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
        "Tomato___Septoria_leaf_spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
picks = [rng.choice([p for p, y in ds.samples if ds.classes[y] == c]) for c in want]
fig, axes = plt.subplots(len(picks), 5, figsize=(11, 2.3 * len(picks)), facecolor=SURFACE)
for row, path in zip(axes, picks):
    color = Image.open(ds.root / path).convert("RGB")
    seg = Image.open(ds.seg_root / Path(path).with_suffix(".png")).convert("RGB")
    ims = [color, seg] + [composite_on_background(seg, random.Random(rng.randint(0, 10**9))) for _ in range(3)]
    titles = ["оригінал", "сегментоване"] + [f"фон {k}" for k in (1, 2, 3)]
    for ax, im, t in zip(row, ims, titles):
        ax.imshow(im); ax.axis("off"); ax.set_title(t, fontsize=8, color=INK)
    row[0].set_title(f"{short_name(path.split('/')[0])}\nоригінал", fontsize=8, color=INK)
fig.suptitle("Аугментація випадковими фонами (v2)", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = REPO / "results" / "v2" / "bg_augmentation.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=90, facecolor=SURFACE)
print("→", out)
