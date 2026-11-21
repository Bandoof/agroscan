"""Датасет і трансформації (спільні для train / eval / Streamlit)."""
import csv
import json
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

REPO = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def load_meta():
    """Класи і тека з фото. Тека: змінна середовища AGROSCAN_IMAGES або шлях з data/classes.json."""
    meta = json.loads((REPO / "data" / "classes.json").read_text(encoding="utf-8"))
    return meta["classes"], Path(os.environ.get("AGROSCAN_IMAGES", meta["images_root"]))


def short_name(cls: str) -> str:
    """'Tomato___Spider_mites Two-spotted_spider_mite' → 'Tomato: Spider mites'."""
    crop, disease = cls.split("___")
    disease = (disease.replace("_", " ").replace("Tomato ", "")
               .replace(" Two-spotted spider mite", "").strip())
    return f"{crop}: {disease}"


def train_transform():
    # Аугментації: геометрія (лист може бути як завгодно повернутий)
    # + колір/яскравість (інше освітлення, ніж у лабораторії).
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PlantVillageSubset(Dataset):
    def __init__(self, split: str, transform=None):
        self.classes, self.root = load_meta()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        with open(REPO / "data" / "splits.csv", encoding="utf-8") as f:
            self.samples = [(r["path"], self.class_to_idx[r["label"]])
                            for r in csv.DictReader(f) if r["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(self.root / path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def targets(self):
        return [y for _, y in self.samples]
