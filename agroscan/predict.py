"""Інференс для демо: фото → топ-k класів з імовірностями (той самий препроцесинг, що й у eval)."""
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data import REPO, eval_transform
from .model import load_checkpoint

DEFAULT_CKPT = REPO / "models" / "agroscan_v2.pt"


class Predictor:
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.classes = ckpt["classes"]
        self.arch = ckpt["arch"]
        self.epoch = ckpt["epoch"]
        self.version = ckpt.get("version", "v1")
        self.tf = eval_transform()
        # v2 навчено з logit adjustment; на val (results/v2/potato_healthy_check.json) кращим виявилось
        # передбачення з логітами + log(prior навчального потоку): те саме або краще top-1/F1 і
        # «Potato: healthy» майже зникає з топ-3. Для v1 prior немає — логіти без змін.
        prior = ckpt.get("train_prior")
        self.logit_bias = torch.log(torch.tensor(prior, dtype=torch.float32)).to(self.device) if prior else None

    @torch.inference_mode()
    def predict(self, img: Image.Image, k: int = 3):
        img = ImageOps.exif_transpose(img).convert("RGB")  # фото з телефона: врахувати поворот з EXIF
        x = self.tf(img).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        logits = self.model(x)
        if self.logit_bias is not None:
            logits = logits + self.logit_bias
        probs = logits.softmax(1)[0].cpu()
        ms = (time.perf_counter() - t0) * 1000
        p, i = probs.topk(k)
        return [(self.classes[j], float(q)) for q, j in zip(p, i)], ms
