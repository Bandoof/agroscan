"""Крок 4b: тест на «польових» фото PlantDoc (Singh et al., 2020) — лише спільні класи.

PlantDoc має 28 класів, наша модель — 17 (PlantVillage: яблуня/картопля/томат).
Оцінюємо ТІЛЬКИ ті PlantDoc-класи, що мають однозначний відповідник серед наших 17.
Модель при цьому передбачає з усіх 17 класів (як у реальному застосуванні),
тож передбачення класу, якого немає в PlantDoc, — це помилка.

Результати в results/:
- plantdoc_class_mapping.csv  — повна таблиця відповідності (28 PlantDoc + 3 наших без пари);
- plantdoc_metrics.json, plantdoc_classification_report.csv;
- confusion_matrix_plantdoc.png, plantdoc_examples.png.

Запуск:
  python scripts/eval_plantdoc.py --zip C:/Users/kuril/datasets/plantdoc/download/plantdoc.zip \
                                  --out C:/Users/kuril/datasets/plantdoc/images
"""
import argparse
import csv
import json
import random
import re
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, eval_transform, short_name  # noqa: E402
from agroscan.model import load_checkpoint  # noqa: E402
from agroscan.plots import INK, SURFACE, confusion_plot  # noqa: E402

RESULTS = REPO / "results"

# PlantDoc-клас → клас PlantVillage (None = відповідника серед наших 17 немає)
MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    # інші культури — поза межами моделі
    "Bell_pepper leaf spot": None, "Bell_pepper leaf": None, "Blueberry leaf": None, "Cherry leaf": None,
    "Corn Gray leaf spot": None, "Corn leaf blight": None, "Corn rust leaf": None, "Peach leaf": None,
    "Raspberry leaf": None, "Soyabean leaf": None, "Squash Powdery mildew leaf": None,
    "Strawberry leaf": None, "grape leaf black rot": None, "grape leaf": None,
}
MEMBER_RE = re.compile(r"^[^/]+/(train|test)/([^/]+)/([^/]+\.(?:jpe?g|png))$", re.IGNORECASE)
UNSAFE_CHARS = re.compile(r"[^\w\-. ()]")  # імена в PlantDoc узяті з URL: %20, ?, +, &, = …


def extract(zip_path, out_dir):
    """Витягує лише класи зі спільним відповідником; повертає [(pd_split, pd_class, path)]."""
    items, counts = [], {}
    with zipfile.ZipFile(zip_path) as zf:
        for k, info in enumerate(zf.infolist()):
            m = MEMBER_RE.match(info.filename)
            if not m:
                continue
            split, cls, fname = m.groups()
            counts[(cls, split)] = counts.get((cls, split), 0) + 1
            if MAPPING.get(cls) is None:  # клас перевірено за білим списком MAPPING
                continue
            # ім'я файлу не відкидаємо, а очищаємо: лише безпечні символи, без шляхів
            # префікс — номер у zip: унікальний навіть якщо очищені імена збіглися
            safe_fname = f"{k:05d}_" + UNSAFE_CHARS.sub("_", fname).lstrip(".")[:150]
            target = out_dir / split / cls / safe_fname
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
            items.append((split, cls, target))
    return items, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=REPO / "models" / "agroscan_efficientnet_b0.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    classes = ckpt["classes"]
    cidx = {c: i for i, c in enumerate(classes)}

    items, counts = extract(args.zip, args.out)
    expected = sum(n for (c, _), n in counts.items() if MAPPING.get(c))
    print(f"Витягнуто фото спільних класів: {len(items)} з {expected} в архіві")
    assert len({str(p) for _, _, p in items}) == len(items), "колізія імен після очищення"

    # --- таблиця відповідності ---
    pd_classes = sorted({c for c, _ in counts}, key=str.lower)
    unknown = [c for c in pd_classes if c not in MAPPING]
    assert not unknown, f"Немає у MAPPING: {unknown}"
    rows = []
    for c in pd_classes:
        pv = MAPPING[c]
        rows.append({"plantdoc_class": c, "plantvillage_class": pv or "—",
                     "n_train": counts.get((c, "train"), 0), "n_test": counts.get((c, "test"), 0),
                     "used": "так" if pv else "ні (немає в моделі)"})
    used_pv = {v for v in MAPPING.values() if v}
    for c in classes:
        if c not in used_pv:
            rows.append({"plantdoc_class": "—", "plantvillage_class": c, "n_train": 0, "n_test": 0,
                         "used": "ні (немає в PlantDoc)"})
    with open(RESULTS / "plantdoc_class_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("Таблиця відповідності PlantDoc → PlantVillage:")
    print(f"  {'PlantDoc':38s} {'PlantVillage (наша модель)':48s} {'train':>5s} {'test':>5s}  використано")
    for r in rows:
        print(f"  {r['plantdoc_class']:38s} {r['plantvillage_class']:48s} {r['n_train']:5d} {r['n_test']:5d}  {r['used']}")

    # --- інференс (усі фото спільних класів: модель їх ніколи не бачила) ---
    tf = eval_transform()
    probs, y_true, splits, paths, bad = [], [], [], [], []
    with torch.no_grad():
        for split, cls, path in items:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:  # noqa: BLE001 — у PlantDoc трапляються биті файли
                bad.append(str(path)); continue
            x = tf(img).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                p = model(x).float().softmax(1).cpu()[0]
            probs.append(p); y_true.append(cidx[MAPPING[cls]]); splits.append(split); paths.append(path)
    P = torch.stack(probs).numpy(); yt = np.array(y_true); pred = P.argmax(1)
    top3 = np.argsort(-P, 1)[:, :3]
    present = sorted(set(yt))
    crop_of = lambda i: classes[i].split("___")[0]  # noqa: E731

    def block(mask):
        yt_, pr_ = yt[mask], pred[mask]
        pm, rm, fm, _ = precision_recall_fscore_support(yt_, pr_, labels=present, average="macro", zero_division=0)
        return {"n": int(mask.sum()), "accuracy": accuracy_score(yt_, pr_),
                "top3_accuracy": float(np.mean([t in row for t, row in zip(yt_, top3[mask])])),
                "crop_accuracy": float(np.mean([crop_of(a) == crop_of(b) for a, b in zip(yt_, pr_)])),
                "precision_macro": pm, "recall_macro": rm, "f1_macro": fm}

    sp = np.array(splits)
    metrics = {"n_common_classes": len(present), "unreadable_files": bad,
               "all_images": block(np.ones(len(yt), bool)), "plantdoc_test_folder_only": block(sp == "test")}
    (RESULTS / "plantdoc_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    pc, rc, fc, sc = precision_recall_fscore_support(yt, pred, labels=present, zero_division=0)
    with open(RESULTS / "plantdoc_classification_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class", "precision", "recall", "f1", "support"])
        for i, c in enumerate(present):
            w.writerow([classes[c], f"{pc[i]:.4f}", f"{rc[i]:.4f}", f"{fc[i]:.4f}", int(sc[i])])

    cm = confusion_matrix(yt, pred, labels=range(len(classes)))[present]
    confusion_plot(cm, [short_name(classes[i]) for i in present], [short_name(c) for c in classes],
                   f"Матриця помилок, PlantDoc (польові фото, {len(present)} спільних класів, n={len(yt)})",
                   RESULTS / "confusion_matrix_plantdoc.png")

    # --- кілька польових фото з передбаченнями ---
    rng = random.Random(42)
    pick = rng.sample(range(len(yt)), 12)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.5), facecolor=SURFACE)
    for ax, i in zip(axes.flat, pick):
        ax.imshow(Image.open(paths[i]).convert("RGB")); ax.axis("off")
        ok = pred[i] == yt[i]
        ax.set_title(f"Справжній: {short_name(classes[yt[i]])}\n"
                     f"Модель: {short_name(classes[pred[i]])} ({P[i, pred[i]]:.0%})\n"
                     f"{'✓ вірно' if ok else '✗ помилка'}", fontsize=8.5, color=INK)
    fig.suptitle("PlantDoc: випадкові 12 польових фото (seed 42) і передбачення моделі", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULTS / "plantdoc_examples.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)

    for name, m in [("усі фото спільних класів", metrics["all_images"]),
                    ("лише тека test PlantDoc", metrics["plantdoc_test_folder_only"])]:
        print(f"\nPlantDoc, {name} (n={m['n']}): accuracy {m['accuracy']:.4f} | top-3 {m['top3_accuracy']:.4f} | "
              f"культура вгадана {m['crop_accuracy']:.4f} | macro P {m['precision_macro']:.4f} "
              f"R {m['recall_macro']:.4f} F1 {m['f1_macro']:.4f}")
    print("\nПо класах:")
    for i, c in enumerate(present):
        print(f"  {short_name(classes[c]):32s} P {pc[i]:.3f}  R {rc[i]:.3f}  F1 {fc[i]:.3f}  n={int(sc[i])}")
    if bad:
        print(f"\nНечитабельних файлів пропущено: {len(bad)}")


if __name__ == "__main__":
    main()
