"""Крок 2a: підготовка підмножини PlantVillage (томат / картопля / яблуня).

1. Із data.zip (mohanty/PlantVillage на Hugging Face) витягує лише кольорові
   зображення класів Apple___*, Potato___*, Tomato___*.
2. Для кожного фото визначає leaf_id (фізичний листок) за leaf-map.json —
   тією ж логікою, що й офіційний скрипт датасету (plant_village.py).
3. Ділить 70/15/15 з фіксованим seed ПО ЛИСТКАХ усередині кожного класу:
   усі фото одного листка потрапляють лише в одну вибірку (без витоку даних),
   а пропорції класів зберігаються (стратифікація).
4. Зберігає data/splits.csv і data/classes.json у репозиторії (без самих фото).

Запуск:
    python scripts/prepare_data.py --download C:/Users/kuril/datasets/plantvillage/download \
                                   --out C:/Users/kuril/datasets/plantvillage/agroscan
"""
import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CROPS = ("Apple", "Potato", "Tomato")
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
# .../color/<Crop___Disease>/<file.jpg>
MEMBER_RE = re.compile(r"(?:^|/)color/((?:%s)___[^/]+)/([^/]+\.(?:jpe?g|png))$" % "|".join(CROPS),
                       re.IGNORECASE)
SAFE_NAME_RE = re.compile(r"^[\w\-. ()]+$")


def leaf_id_for(class_name: str, file_name: str, leaf_map: dict) -> str:
    """Повторює логіку визначення leaf_id з офіційного plant_village.py."""
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()
    suggestions = leaf_map.get(ident.lower().strip())
    if suggestions:
        if len(suggestions) == 1:
            return suggestions[0]
        for s in suggestions:
            if class_name in s:
                return s
    return f"fallback_{class_name}_{ident}"


def extract(zip_path: Path, images_dir: Path) -> list[tuple[str, str]]:
    """Безпечно витягує потрібні файли; повертає [(class_name, file_name)]."""
    items = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            m = MEMBER_RE.search(info.filename)
            if not m or info.is_dir():
                continue
            class_name, file_name = m.group(1), m.group(2)
            # захист від path traversal: лише «прості» імена, пишемо за власним шляхом
            if not (SAFE_NAME_RE.match(class_name) and SAFE_NAME_RE.match(file_name)):
                print(f"  пропущено підозріле ім'я: {info.filename!r}")
                continue
            target = images_dir / class_name / file_name
            if not target.exists() or target.stat().st_size != info.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            items.append((class_name, file_name))
    return items


def split_by_leaf(items, leaf_map, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(lambda: defaultdict(list))  # class -> leaf -> [files]
    for cls, fname in items:
        by_class[cls][leaf_id_for(cls, fname, leaf_map)].append(fname)

    rows = []
    for cls in sorted(by_class):
        leaves = sorted(by_class[cls])
        rng.shuffle(leaves)
        n = len(leaves)
        n_train = round(n * SPLITS["train"])
        n_val = round(n * SPLITS["val"])
        parts = {"train": leaves[:n_train],
                 "val": leaves[n_train:n_train + n_val],
                 "test": leaves[n_train + n_val:]}
        for split, split_leaves in parts.items():
            for leaf in split_leaves:
                for fname in sorted(by_class[cls][leaf]):
                    rows.append({"path": f"{cls}/{fname}", "label": cls,
                                 "leaf_id": leaf, "split": split})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=Path, required=True, help="тека з data.zip і leaf-map.json")
    ap.add_argument("--out", type=Path, required=True, help="куди класти зображення")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    images_dir = args.out / "images"
    leaf_map = json.loads((args.download / "leaf-map.json").read_text(encoding="utf-8"))

    print("Розпаковую потрібні класи…")
    items = extract(args.download / "data.zip", images_dir)
    print(f"  зображень: {len(items)}")

    rows = split_by_leaf(items, leaf_map, args.seed)

    # Перевірка витоку: жоден листок не може бути у двох вибірках
    leaf_splits = defaultdict(set)
    for r in rows:
        leaf_splits[r["leaf_id"]].add(r["split"])
    leaked = [l for l, s in leaf_splits.items() if len(s) > 1]
    assert not leaked, f"Витік: {len(leaked)} листків у кількох вибірках"
    n_fallback = sum(1 for l in leaf_splits if l.startswith("fallback_"))

    classes = sorted({r["label"] for r in rows})
    (repo / "data").mkdir(exist_ok=True)
    with open(repo / "data" / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "leaf_id", "split"])
        w.writeheader()
        w.writerows(rows)
    (repo / "data" / "classes.json").write_text(
        json.dumps({"classes": classes, "seed": args.seed, "images_root": str(images_dir)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    cnt = Counter(r["split"] for r in rows)
    total = len(rows)
    print(f"Класів: {len(classes)}, зображень: {total}, унікальних листків: {len(leaf_splits)}"
          f" (без збігу в leaf-map: {n_fallback})")
    for s in SPLITS:
        print(f"  {s:5s}: {cnt[s]:6d} ({cnt[s] / total:.1%})")
    print("Витоку між вибірками немає (перевірено по leaf_id).")


if __name__ == "__main__":
    main()
