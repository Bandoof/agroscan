"""Перевірка демо-логіки без UI: по одному тестовому фото кожного класу → топ-3 + наявність поради."""
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.advice import ADVICE, NAME_UA  # noqa: E402
from agroscan.data import REPO, load_meta, short_name  # noqa: E402
from agroscan.predict import Predictor  # noqa: E402

pred = Predictor(device="cpu")
assert set(ADVICE) == set(pred.classes), set(pred.classes) ^ set(ADVICE)
assert set(NAME_UA) == set(pred.classes), set(pred.classes) ^ set(NAME_UA)
_, root = load_meta()
with open(REPO / "data" / "splits.csv", encoding="utf-8") as f:
    first = {}
    for r in csv.DictReader(f):
        if r["split"] == "test":
            first.setdefault(r["label"], r["path"])
ok = 0
for cls, path in sorted(first.items()):
    top3, ms = pred.predict(Image.open(root / path))
    hit = top3[0][0] == cls
    ok += hit
    print(f"{'✓' if hit else '✗'} {short_name(cls):30s} → " +
          " | ".join(f"{short_name(c)} {p:.0%}" for c, p in top3) + f"  ({ms:.0f} мс)")
print(f"{ok}/{len(first)} вірних; поради є для всіх {len(ADVICE)} класів")
