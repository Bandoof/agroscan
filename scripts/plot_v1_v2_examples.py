"""Рисунок results/v2/v1_vs_v2_examples.png: приклади з PlantDoc test без витоку (107 фото).

- 8 фото, де v2 правильна, а v1 ні; 2 фото, де навпаки (v1 правильна, v2 ні).
- Фото обираються ВИПАДКОВО (seed 42) з відповідних груп, а не вручну, щоб не підбирати «красиві» приклади.
- Передбачення — той самий конвеєр, що й у compare_v1_v2.py (AMP FP16; v2 — логіти + log(prior)).
"""
import json
import random
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agroscan.advice import NAME_UA  # noqa: E402
from agroscan.data import REPO, eval_transform  # noqa: E402
from agroscan.data_v2 import PlantDocSubset  # noqa: E402
from agroscan.model import load_checkpoint  # noqa: E402
from agroscan.plots import INK, INK2, SURFACE  # noqa: E402
from compare_v1_v2 import logits_for  # noqa: E402

N_V2_WINS, N_V1_WINS, SEED = 8, 2, 42


def main():
    device = torch.device("cuda")
    ds = PlantDocSubset("pd_test", eval_transform())
    leaked = {l["test"] for l in json.loads((REPO / "data" / "plantdoc_test_leaks.json")
                                            .read_text(encoding="utf-8"))["leaked"]}
    clean = np.array([p not in leaked for p, _ in ds.samples])

    P = {}
    for ver in ("v1", "v2"):
        model, ckpt = load_checkpoint(REPO / "models" / f"agroscan_{ver}.pt", device)
        L, y = logits_for(model, ds, device)
        if ver == "v2":
            L = L + torch.log(torch.tensor(ckpt["train_prior"]))
        P[ver] = L.softmax(1).numpy()
        classes = ckpt["classes"]
        del model; torch.cuda.empty_cache()

    p1, p2 = P["v1"].argmax(1), P["v2"].argmax(1)
    idx = np.nonzero(clean)[0]
    v2_wins = [int(i) for i in idx if p2[i] == y[i] and p1[i] != y[i]]
    v1_wins = [int(i) for i in idx if p1[i] == y[i] and p2[i] != y[i]]
    print(f"Чистий test: {clean.sum()} фото; v2 вірно, v1 ні: {len(v2_wins)}; v1 вірно, v2 ні: {len(v1_wins)}")
    rng = random.Random(SEED)
    pick_v2 = sorted(rng.sample(v2_wins, min(N_V2_WINS, len(v2_wins))))
    pick_v1 = sorted(rng.sample(v1_wins, min(N_V1_WINS, len(v1_wins))))

    def label(i):
        w = lambda s: "\n".join(textwrap.wrap(s, 34))  # noqa: E731
        ok1, ok2 = p1[i] == y[i], p2[i] == y[i]
        return (f"Справжній: {w(NAME_UA[classes[y[i]]])}\n"
                f"v1: {w(NAME_UA[classes[p1[i]]])} ({P['v1'][i, p1[i]]:.0%}) {'✓' if ok1 else '✗'}\n"
                f"v2: {w(NAME_UA[classes[p2[i]]])} ({P['v2'][i, p2[i]]:.0%}) {'✓' if ok2 else '✗'}")

    ncol = 4
    rows = [pick_v2[k:k + ncol] for k in range(0, len(pick_v2), ncol)] + [pick_v1]
    fig = plt.figure(figsize=(ncol * 3.6, len(rows) * 4.3 + 1.0), facecolor=SURFACE)
    gs = fig.add_gridspec(len(rows), ncol, hspace=0.62, wspace=0.12, top=0.90, bottom=0.10, left=0.02, right=0.98)
    for r, row in enumerate(rows):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c]); ax.axis("off")
            if c < len(row):
                i = row[c]
                ax.imshow(Image.open(ds.root / ds.samples[i][0]).convert("RGB"))
                ax.text(0.0, -0.03, label(i), transform=ax.transAxes, va="top", ha="left", fontsize=9,
                        color=INK, linespacing=1.25)
    # підписи груп
    fig.text(0.02, 0.955, f"v2 правильна, v1 помиляється — {len(pick_v2)} з {len(v2_wins)} таких фото "
             f"(випадковий вибір, seed {SEED})", fontsize=11, color=INK, weight="bold")
    y_v1 = gs[len(rows) - 1, 0].get_position(fig).y1 + 0.012
    fig.text(0.02, y_v1, f"v1 правильна, v2 помиляється — {len(pick_v1)} з {len(v1_wins)}",
             fontsize=11, color=INK, weight="bold")
    fig.suptitle(f"PlantDoc test без витоку (n = {int(clean.sum())}): приклади, де моделі розходяться",
                 x=0.02, ha="left", y=0.995, fontsize=13, color=INK)
    fig.text(0.98, 0.955, "v2 — з log(prior); у дужках — впевненість top-1", ha="right", fontsize=8.5, color=INK2)
    out = REPO / "results" / "v2" / "v1_vs_v2_examples.png"
    fig.savefig(out, dpi=95, facecolor=SURFACE)
    print("→", out, "| вибрано:", [ds.samples[i][0] for i in pick_v2 + pick_v1])


if __name__ == "__main__":
    main()
