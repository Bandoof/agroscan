"""Розподіл top-1 впевненості моделі: PlantVillage test vs PlantDoc, вірні vs помилкові.
Потрібно, щоб поріг «низька впевненість» у демо був обґрунтований даними, а не вигаданий.
Результат → results/confidence_check.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, load_meta  # noqa: E402
from agroscan.predict import Predictor  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from eval_plantdoc import MAPPING  # noqa: E402

PD_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/kuril/datasets/plantdoc/images")


def collect(pairs, pred):
    conf, ok = [], []
    for path, label in pairs:
        top, _ms = pred.predict(Image.open(path), k=1)
        c, p = top[0]
        conf.append(p); ok.append(c == label)
    return np.array(conf), np.array(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    pred = Predictor(REPO / "models" / f"agroscan_{args.version}.pt", device="cuda")
    _, root = load_meta()
    if args.version == "v1":
        # v1 не бачив PlantDoc → можна брати всі фото спільних класів; PlantVillage — test
        with open(REPO / "data" / "splits.csv", encoding="utf-8") as f:
            pv = [(root / r["path"], r["label"]) for r in csv.DictReader(f) if r["split"] == "test"]
        pd = [(p, MAPPING[p.parent.name]) for p in PD_ROOT.rglob("*") if p.is_file() and MAPPING.get(p.parent.name)]
        sets, out_path = [("plantvillage_test", pv), ("plantdoc", pd)], REPO / "results" / "confidence_check.json"
    else:
        # v2 вчився на PlantDoc train → поріг обираємо лише на VAL (PlantVillage val, PlantDoc pd_val)
        from agroscan.data_v2 import load_plantdoc_root
        pd_root = load_plantdoc_root()
        with open(REPO / "data" / "splits.csv", encoding="utf-8") as f:
            pv = [(root / r["path"], r["label"]) for r in csv.DictReader(f) if r["split"] == "val"]
        with open(REPO / "data" / "plantdoc_splits.csv", encoding="utf-8") as f:
            pd = [(pd_root / r["path"], r["label"]) for r in csv.DictReader(f) if r["split"] == "pd_val"]
        sets, out_path = [("plantvillage_val", pv), ("plantdoc_val", pd)], REPO / "results" / "v2" / "confidence_check_v2.json"
    out = {}
    for name, pairs in sets:
        conf, ok = collect(pairs, pred)
        q = lambda a: {f"p{k}": float(np.percentile(a, k)) for k in (10, 25, 50, 75, 90)} if len(a) else {}  # noqa: E731
        out[name] = {"n": len(conf), "accuracy": float(ok.mean()), "correct": q(conf[ok]), "wrong": q(conf[~ok])}
        for t in (0.4, 0.5, 0.6, 0.7):
            m = conf >= t
            out[name][f"thr_{t}"] = {"share_above": float(m.mean()),
                                     "accuracy_above": float(ok[m].mean()) if m.any() else None,
                                     "accuracy_below": float(ok[~m].mean()) if (~m).any() else None}
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, d in out.items():
        print(f"{name} (n={d['n']}, acc {d['accuracy']:.3f})")
        print("  медіана впевненості: вірні {:.3f}, помилкові {:.3f}".format(
            d["correct"].get("p50", float("nan")), d["wrong"].get("p50", float("nan"))))
        for t in (0.4, 0.5, 0.6, 0.7):
            r = d[f"thr_{t}"]
            print(f"  поріг {t}: вище порогу {r['share_above']:.1%} фото, точність вище {r['accuracy_above']}, "
                  f"нижче {r['accuracy_below']}")


if __name__ == "__main__":
    main()
