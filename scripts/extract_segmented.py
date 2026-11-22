"""Крок 4б-2: витягує сегментовані (фон прибрано) версії TRAIN-фото PlantVillage для аугментації фонами.

Зіставлення: raw/color/<cls>/<stem>.JPG  ↔  raw/segmented/<cls>/<stem>_final_masked.jpg
Зберігає як <out>/<cls>/<stem>.png (PNG без втрат, щоб маска фону не «плила» від JPEG).
Лише train — val/test не потрібні (аугментація тільки під час навчання).

Запуск: python scripts/extract_segmented.py --zip <data.zip> --out C:/Users/kuril/datasets/plantvillage/agroscan/segmented
"""
import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO  # noqa: E402

SEG_RE = re.compile(r"^raw/segmented/((?:Apple|Potato|Tomato)___[^/]+)/([^/]+)_final_masked\.(?:jpe?g|png)$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(REPO / "data" / "splits.csv", encoding="utf-8") as f:
        wanted = {}
        for r in csv.DictReader(f):
            if r["split"] == "train":
                cls, fn = r["path"].split("/")
                wanted[(cls, fn.rsplit(".", 1)[0].lower())] = (cls, fn.rsplit(".", 1)[0])
    done = 0
    with zipfile.ZipFile(args.zip) as zf:
        for info in zf.infolist():
            m = SEG_RE.match(info.filename)
            if not m:
                continue
            key = (m.group(1), m.group(2).lower())
            if key not in wanted:
                continue
            cls, stem = wanted[key]
            target = args.out / cls / f"{stem}.png"  # ім'я з нашого splits.csv, не з архіву
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.open(io.BytesIO(zf.read(info))).convert("RGB").save(target)
            done += 1
    meta_p = REPO / "data" / "classes.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["segmented_root"] = str(args.out)
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сегментованих train-фото: {done} з {len(wanted)}")


if __name__ == "__main__":
    main()
