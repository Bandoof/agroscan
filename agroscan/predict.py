"""Інференс для демо: фото → топ-k класів з імовірностями (той самий препроцесинг, що й у eval)."""
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data import REPO, eval_transform
from .model import load_checkpoint

DEFAULT_CKPT = REPO / "models" / "agroscan_efficientnet_b0.pt"


class Predictor:
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.classes = ckpt["classes"]
        self.arch = ckpt["arch"]
        self.epoch = ckpt["epoch"]
        self.tf = eval_transform()

    @torch.inference_mode()
    def predict(self, img: Image.Image, k: int = 3):
        img = ImageOps.exif_transpose(img).convert("RGB")  # фото з телефона: врахувати поворот з EXIF
        x = self.tf(img).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        probs = self.model(x).softmax(1)[0].cpu()
        ms = (time.perf_counter() - t0) * 1000
        p, i = probs.topk(k)
        return [(self.classes[j], float(q)) for q, j in zip(p, i)], ms
