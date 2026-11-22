"""Фіксує ручний вердикт по кандидатах із results/v2/leak_check.json → data/plantdoc_test_leaks.json.

Кожну пару «фото test ↔ найближче навчальне фото» переглянуто очима (results/v2/leak_candidates.png,
великі фото — окремо). LEAKED — індекси кандидатів, де це ТЕ САМЕ фото (кроп / інше стиснення / водяний знак).
Решта кандидатів — різні фото схожої композиції.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agroscan.data import REPO  # noqa: E402

LEAKED = {
    0: "те саме стокове фото (alamy), інший водяний знак",
    1: "те саме фото, інший кроп і масштаб",
    2: "test — кроп навчального фото; мітки різні (Tomato EB vs Potato EB)",
    5: "те саме фото, інший кроп",
    6: "те саме фото (рука з тим самим листком); мітки різні (Bacterial spot vs Septoria)",
    9: "те саме стокове фото (alamy), інший кроп і водяний знак",
    10: "test — кроп верхньої частини навчального фото",
}

rep = json.loads((REPO / "results" / "v2" / "leak_check.json").read_text(encoding="utf-8"))
cands = rep["candidates"]
leaks = [{"test": cands[k]["test"], "test_label": cands[k]["test_label"], "train_pool": cands[k]["pool"],
          "train_match": cands[k]["match"], "train_label": cands[k]["match_label"], "cos": cands[k]["cos"],
          "note": note} for k, note in sorted(LEAKED.items())]
out = {"method": "EfficientNet-B0 (ImageNet) cos >= 0.78 або dHash <= 12, потім ручний перегляд кожної пари",
       "n_candidates_reviewed": len(cands), "n_leaked_test_photos": len(leaks), "leaked": leaks}
(REPO / "data" / "plantdoc_test_leaks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
for l in leaks:
    print(f"{l['cos']:.3f} {l['train_pool']:8s} {l['test_label']:40s} | {l['note']}")
print(f"Виключити з test: {len(leaks)} з {rep['n_test']}")
