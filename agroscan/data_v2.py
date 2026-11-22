"""Дані для моделі v2: PlantVillage + PlantDoc, випадкові фони, збалансований семплер.

- PlantDocSubset: data/plantdoc_splits.csv (pd_train / pd_val / pd_test).
- PlantVillageBG: train-фото PlantVillage; з імовірністю p_bg замість кольорового фото береться
  сегментоване (фон чорний) і листок кладеться на випадковий синтетичний фон.
- make_sampler: частка PlantDoc у кожній епосі = pd_share; усередині джерела класи
  семплюються з вагою ∝ 1/sqrt(n_класу). Повертає також фактичний апріорний розподіл класів
  у навчальному потоці — його використовує logit adjustment у loss.
"""
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.utils.data import Dataset, WeightedRandomSampler

from .data import REPO, PlantVillageSubset, load_meta


def load_plantdoc_root() -> Path:
    meta = json.loads((REPO / "data" / "plantdoc_meta.json").read_text(encoding="utf-8"))
    return Path(os.environ.get("AGROSCAN_PLANTDOC", meta["images_root"]))


class PlantDocSubset(Dataset):
    def __init__(self, split: str, transform=None):
        classes, _ = load_meta()
        self.classes = classes
        idx = {c: i for i, c in enumerate(classes)}
        self.root = load_plantdoc_root()
        with open(REPO / "data" / "plantdoc_splits.csv", encoding="utf-8") as f:
            self.samples = [(r["path"], idx[r["label"]]) for r in csv.DictReader(f) if r["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(self.root / path).convert("RGB")
        return (self.transform(img) if self.transform else img), y

    def targets(self):
        return [y for _, y in self.samples]


# ---------- випадкові фони ----------
def _rand_color(rng):
    return np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)


def random_background(w, h, rng: random.Random) -> np.ndarray:
    kind = rng.choice(["solid", "gradient", "noise", "noise"])
    if kind == "solid":
        bg = np.ones((h, w, 3), np.float32) * _rand_color(rng)
    elif kind == "gradient":
        c1, c2 = _rand_color(rng), _rand_color(rng)
        horizontal = rng.random() < 0.5
        t = np.linspace(0, 1, w if horizontal else h, dtype=np.float32)
        t = t[None, :, None] if horizontal else t[:, None, None]
        bg = c1 * (1 - t) + c2 * t
        bg = np.broadcast_to(bg, (h, w, 3)).copy()
    else:  # плямиста «ґрунт/трава»-подібна текстура: низькочастотний шум, розтягнутий до розміру
        s = rng.choice([4, 8, 16, 32])
        base = np.stack([np.random.default_rng(rng.randint(0, 2**31)).random((s, s)) for _ in range(3)], -1)
        base = base * rng.uniform(80, 200) + _rand_color(rng) * rng.uniform(0.2, 0.7)
        bg = np.asarray(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                        np.float32)
    bg += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0, 12), bg.shape)
    return np.clip(bg, 0, 255)


def leaf_mask(seg: Image.Image, thr: int = 25) -> np.ndarray:
    """Маска листка з сегментованого фото (фон ~чорний). Дірки всередині (темні плями хвороби)
    заповнюються, дрібне сміття прибирається; краї трохи розмиваються для природного переходу."""
    a = np.asarray(seg, np.int16)
    m = a.max(-1) > thr
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    soft = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    return np.asarray(soft, np.float32)[..., None] / 255.0


def composite_on_background(seg: Image.Image, rng: random.Random) -> Image.Image:
    m = leaf_mask(seg)
    bg = random_background(seg.width, seg.height, rng)
    out = np.asarray(seg, np.float32) * m + bg * (1 - m)
    return Image.fromarray(out.astype(np.uint8))


class PlantVillageBG(PlantVillageSubset):
    """Train-вибірка PlantVillage з аугментацією випадковими фонами (лише для навчання)."""

    def __init__(self, transform=None, p_bg: float = 0.5):
        super().__init__("train", transform)
        meta = json.loads((REPO / "data" / "classes.json").read_text(encoding="utf-8"))
        self.seg_root = Path(os.environ.get("AGROSCAN_SEGMENTED", meta["segmented_root"]))
        self.p_bg = p_bg

    def __getitem__(self, i):
        path, y = self.samples[i]
        seg_path = self.seg_root / Path(path).with_suffix(".png")
        rng = random.Random()  # свій генератор: різний у кожному воркері
        if self.p_bg > 0 and rng.random() < self.p_bg and seg_path.exists():
            img = composite_on_background(Image.open(seg_path).convert("RGB"), rng)
        else:
            img = Image.open(self.root / path).convert("RGB")
        return (self.transform(img) if self.transform else img), y


def make_sampler(pv_targets, pd_targets, num_classes, pd_share=0.25, num_samples=None, seed=42):
    """Ваги для ConcatDataset([PV, PD]). Повертає (sampler, class_prior[num_classes])."""
    weights, prior = [], np.zeros(num_classes)
    for targets, share in ((pv_targets, 1 - pd_share), (pd_targets, pd_share)):
        cnt = Counter(targets)
        z = sum(np.sqrt(n) for n in cnt.values())  # Σ_y n_y · n_y^(-1/2)
        weights += [share * cnt[y] ** -0.5 / z for y in targets]
        for y, n in cnt.items():
            prior[y] += share * np.sqrt(n) / z
    n = num_samples or (len(pv_targets) + len(pd_targets))
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), n, replacement=True, generator=g)
    return sampler, prior / prior.sum()
