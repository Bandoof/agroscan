"""Крок 4б: порівняння v1 і v2 на PlantVillage test (3463) і PlantDoc test (114).

Обидві моделі проходять ОДИН і той самий конвеєр (eval_transform, AMP FP16, ті самі фото).
Метрики: top-1, top-3, macro-F1 (PlantDoc — по класах, присутніх у test).
PlantDoc test малий, тому:
- bootstrap 95 % довірчі інтервали (2000 повторів, seed 0) для кожної моделі;
- парний bootstrap для різниці v2 − v1 на тих самих фото + точний тест Мак-Немара.
Окремо: як часто «Potato: healthy» у топ-3, коли це НЕ справжній клас.

Результати: results/comparison_v1_v2.md, results/v2/comparison_v1_v2.json, results/v2/confusion_matrix_plantdoc_test_v2.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO, PlantVillageSubset, eval_transform, short_name  # noqa: E402
from agroscan.data_v2 import PlantDocSubset  # noqa: E402
from agroscan.model import load_checkpoint  # noqa: E402
from agroscan.plots import confusion_plot  # noqa: E402

B, SEED = 2000, 0


@torch.no_grad()
def logits_for(model, ds, device):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
    out, ys = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys).numpy()


def metrics(P, y, labels, ph_idx):
    top3 = np.argsort(-P, 1)[:, :3]
    pred = top3[:, 0]
    not_ph = y != ph_idx
    return {"n": int(len(y)), "top1": float((pred == y).mean()),
            "top3": float(np.mean([t in r for t, r in zip(y, top3)])),
            "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "potato_healthy_in_top3_when_not_true": float(np.mean([ph_idx in r for r in top3[not_ph]]))}


def boot_ci(P1, P2, y, labels):
    """95 % інтервали для кожної моделі та для різниці v2 − v1 (парний bootstrap)."""
    rng = np.random.default_rng(SEED)
    pred1, pred2 = P1.argmax(1), P2.argmax(1)
    t3_1 = np.array([t in r for t, r in zip(y, np.argsort(-P1, 1)[:, :3])])
    t3_2 = np.array([t in r for t, r in zip(y, np.argsort(-P2, 1)[:, :3])])
    res = {k: [] for k in ("top1_v1", "top1_v2", "top1_diff", "top3_v1", "top3_v2", "top3_diff",
                           "f1_v1", "f1_v2", "f1_diff")}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        a1, a2 = (pred1[i] == y[i]).mean(), (pred2[i] == y[i]).mean()
        f1a = f1_score(y[i], pred1[i], labels=labels, average="macro", zero_division=0)
        f1b = f1_score(y[i], pred2[i], labels=labels, average="macro", zero_division=0)
        for k, v in (("top1_v1", a1), ("top1_v2", a2), ("top1_diff", a2 - a1),
                     ("top3_v1", t3_1[i].mean()), ("top3_v2", t3_2[i].mean()),
                     ("top3_diff", t3_2[i].mean() - t3_1[i].mean()),
                     ("f1_v1", f1a), ("f1_v2", f1b), ("f1_diff", f1b - f1a)):
            res[k].append(v)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}
    # Мак-Немар (точний): фото, де вгадала лише одна з моделей
    only2 = int(((pred2 == y) & (pred1 != y)).sum()); only1 = int(((pred1 == y) & (pred2 != y)).sum())
    p = binomtest(only2, only2 + only1, 0.5).pvalue if only1 + only2 else 1.0
    return ci, {"only_v2_correct": only2, "only_v1_correct": only1, "mcnemar_exact_p": float(p)}


def label_conflicts():
    """Скільки фото test мають копію в train (виключену з навчання) з ІНШОЮ міткою."""
    import csv
    from collections import defaultdict
    with open(REPO / "data" / "plantdoc_splits.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    g = defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)
    pairs, conf, photos = 0, 0, set()
    for mem in g.values():
        for t in (r for r in mem if r["split"] == "pd_test"):
            for e in (r for r in mem if r["split"] == "pd_excluded_dup"):
                pairs += 1
                if t["pd_class"] != e["pd_class"]:
                    conf += 1; photos.add(t["path"])
    return {"train_test_duplicate_pairs": pairs, "pairs_with_different_labels": conf,
            "test_photos_with_conflicting_copy": len(photos)}


def main():
    device = torch.device("cuda")
    conflicts = label_conflicts()
    pv_test = PlantVillageSubset("test", eval_transform())
    pd_test = PlantDocSubset("pd_test", eval_transform())
    classes = pv_test.classes
    ph = classes.index("Potato___healthy")
    pd_labels = sorted(set(pd_test.targets()))
    # Фото test, що мають копію (той самий знімок / кроп) у навчанні v2 — data/plantdoc_test_leaks.json
    leaks = json.loads((REPO / "data" / "plantdoc_test_leaks.json").read_text(encoding="utf-8"))
    leaked = {l["test"] for l in leaks["leaked"]}
    clean = np.array([p not in leaked for p, _ in pd_test.samples])
    assert (~clean).sum() == len(leaked), "не всі виключені фото знайдено в test"

    # v2_raw — звичайні логіти; v2 — фінальний режим (логіти + log(prior)), обраний на VAL
    results, probs = {}, {}
    for ver in ("v1", "v2"):
        model, ckpt = load_checkpoint(REPO / "models" / f"agroscan_{ver}.pt", device)
        Lpv, ypv = logits_for(model, pv_test, device)
        Lpd, ypd = logits_for(model, pd_test, device)
        modes = {"v1": None} if ver == "v1" else {"v2_raw": None,
                                                 "v2": torch.log(torch.tensor(ckpt["train_prior"]))}
        for name, bias in modes.items():
            b = bias if bias is not None else 0.0
            Ppv, Ppd = (Lpv + b).softmax(1).numpy(), (Lpd + b).softmax(1).numpy()
            probs[name] = (Ppd, ypd)
            results[name] = {"epoch": ckpt["epoch"],
                             "plantvillage_test": metrics(Ppv, ypv, list(range(len(classes))), ph),
                             "plantdoc_test_clean": metrics(Ppd[clean], ypd[clean], pd_labels, ph),
                             "plantdoc_test_full": metrics(Ppd, ypd, pd_labels, ph)}
        del model; torch.cuda.empty_cache()

    ypd = probs["v1"][1]
    ci, mcn = boot_ci(probs["v1"][0][clean], probs["v2"][0][clean], ypd[clean], pd_labels)
    ci_full, mcn_full = boot_ci(probs["v1"][0], probs["v2"][0], ypd, pd_labels)
    pd_f1_per_class = {}
    for ver in ("v1", "v2_raw", "v2"):
        pr = probs[ver][0][clean].argmax(1)
        f = f1_score(ypd[clean], pr, labels=pd_labels, average=None, zero_division=0)
        pd_f1_per_class[ver] = {classes[c]: float(v) for c, v in zip(pd_labels, f)}
    n_clean = int(clean.sum())
    cm = confusion_matrix(ypd[clean], probs["v2"][0][clean].argmax(1), labels=range(len(classes)))[pd_labels]
    out_dir = REPO / "results" / "v2"; out_dir.mkdir(parents=True, exist_ok=True)
    confusion_plot(cm, [short_name(classes[i]) for i in pd_labels], [short_name(c) for c in classes],
                   f"v2: матриця помилок, PlantDoc test без витоку (n={n_clean})",
                   out_dir / "confusion_matrix_plantdoc_test_v2.png", normalize=False)

    summary = {"results": results, "plantdoc_test_clean_n": n_clean, "plantdoc_test_excluded_leaks": sorted(leaked),
               "plantdoc_test_clean_bootstrap_ci95": ci, "plantdoc_test_clean_paired": mcn,
               "plantdoc_test_full_bootstrap_ci95": ci_full, "plantdoc_test_full_paired": mcn_full,
               "plantdoc_test_clean_f1_per_class": pd_f1_per_class, "bootstrap_resamples": B,
               "plantdoc_test_classes": len(pd_labels), "plantdoc_label_conflicts": conflicts}
    (out_dir / "comparison_v1_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    r1, r0, r2 = results["v1"], results["v2_raw"], results["v2"]
    pct = lambda v: f"{v*100:.1f} %"  # noqa: E731
    civ = lambda c, k: f"[{c[k][0]*100:.1f}; {c[k][1]*100:.1f}]"  # noqa: E731
    lines = [
        "# Порівняння моделей v1 і v2", "",
        "Обидві моделі оцінено однаковим конвеєром (Resize 256 → CenterCrop 224, AMP FP16) на тих самих фото.",
        "**v2** — фінальний режим передбачення: логіти + log(prior навчального потоку). Цей режим обрано на "
        "валідації (PlantVillage val і PlantDoc val, `results/v2/potato_healthy_check.json`), не на test. "
        "Для довідки наведено й v2 зі звичайними логітами.", "",
        f"**PlantDoc test без витоку (n={n_clean})** — основний результат: з 114 фото виключено {len(leaked)}, "
        "копії яких (той самий знімок, кроп або інше стиснення) потрапили в навчання v2 "
        "(`data/plantdoc_test_leaks.json`, метод — `scripts/leak_check.py` + ручний перегляд). "
        "Повний test (114) наведено для довідки.", "",
        "| Набір | Метрика | v1 | v2 (звичайні логіти) | **v2** | Різниця v2 − v1 (95 % ДІ для PlantDoc) |",
        "|---|---|---|---|---|---|"]
    for name, key, c in (("PlantVillage test (n=3463)", "plantvillage_test", None),
                         (f"**PlantDoc test без витоку (n={n_clean})**", "plantdoc_test_clean", ci),
                         ("PlantDoc test повний (n=114), довідково", "plantdoc_test_full", ci_full)):
        for m, label in (("top1", "top-1"), ("top3", "top-3"), ("f1_macro", "macro-F1")):
            a, raw, b = r1[key][m], r0[key][m], r2[key][m]
            diff = f"{(b-a)*100:+.1f} п.п."
            if c is not None:
                kk = {"top1": "top1", "top3": "top3", "f1_macro": "f1"}[m]
                diff += f" {civ(c, kk + '_diff')}"
            lines.append(f"| {name} | {label} | {pct(a)} | {pct(raw)} | **{pct(b)}** | {diff} |")
    ph_key = "potato_healthy_in_top3_when_not_true"
    lines += ["", "**«Potato: healthy» у топ-3, коли це не справжній клас:**", "",
              "| Набір | v1 | v2 (звичайні логіти) | **v2** |", "|---|---|---|---|"]
    for name, key in (("PlantVillage test", "plantvillage_test"), (f"PlantDoc test без витоку", "plantdoc_test_clean")):
        lines.append(f"| {name} | {pct(r1[key][ph_key])} | {pct(r0[key][ph_key])} | **{pct(r2[key][ph_key])}** |")
    lines += ["",
              "Logit adjustment сам по собі «Potato: healthy» з топ-3 не прибрав: він навмисно підсилює рідкісні "
              "класи, а цей клас найрідкісніший. Прибрало його додавання log(prior) при передбаченні.", "",
              f"## ⚠️ PlantDoc test малий ({n_clean} фото без витоку)", "",
              f"- 95 % bootstrap-інтервал top-1: v1 {civ(ci, 'top1_v1')}, v2 {civ(ci, 'top1_v2')}; "
              f"macro-F1: v1 {civ(ci, 'f1_v1')}, v2 {civ(ci, 'f1_v2')}.",
              f"- Парне порівняння на тих самих фото: лише v2 вірно — {mcn['only_v2_correct']}, "
              f"лише v1 вірно — {mcn['only_v1_correct']}; точний тест Мак-Немара p = {mcn['mcnemar_exact_p']:.2g}.",
              f"- Macro-F1 рахується по {len(pd_labels)} класах, у більшості класів у test лише 6–11 фото, "
              "тож macro-F1 тут особливо шумний.",
              f"- Витік: {len(leaked)} фото test мали копію в навчанні v2 (5 — у pd_train, 2 — у pd_val, "
              "за якою обирали епоху). Перша перевірка (MD5 + dHash ≤ 4) їх не знайшла, бо це кропи/перестиснення; "
              "знайдено ембедингами ImageNet-моделі й підтверджено вручну.",
              f"- Шум у мітках: {conflicts['train_test_duplicate_pairs']} точних/майже точних пар «train ↔ test» "
              f"виключено ще до навчання; у {conflicts['pairs_with_different_labels']} з них мітки різні "
              f"(зачіпає {conflicts['test_photos_with_conflicting_copy']} фото test). Серед 7 фото з витоком "
              "ще у 2 мітки різні. Неоднозначність міток обмежує досяжну точність на test.",
              "- Епоху v2 обрано за PlantDoc val (224 фото); test у навчанні й виборі моделі не використано.", "",
              f"Сирі дані: [`results/v2/comparison_v1_v2.json`](v2/comparison_v1_v2.json)."]
    (REPO / "results" / "comparison_v1_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
