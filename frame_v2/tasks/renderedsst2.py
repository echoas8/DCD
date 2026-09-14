# frame/tasks/renderedsst2.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class RenderedSST2Task(Task):
    """
    Rendered-SST2 任务�?
    目录结构�?        DATA_ROOT = "./datasets/datasets/renderedsst2/rendered-sst2"
        DATA_ROOT/<split>/<class_name>/*.png|*.jpg|...

    """

    key = "renderedsst2"

    def __init__(self, root: str | None = None, split: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "REND_SST2_ROOT",
            root or "./datasets/datasets/renderedsst2/rendered-sst2"
        ))

        # split：环境变�?> 参数 > 默认 "test"
        self.split = os.environ.get("REND_SST2_SPLIT", split or "test").lower()

        # 类别名和 label 映射（和你原脚本保持一致）
        self.class_to_label = {"negative": 0, "positive": 1}
        self._samples: list[Path] = []
        self._load_samples()

    def _load_samples(self):
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"[RenderedSST2] Split dir not found: {split_dir}")

        samples: list[Path] = []
        for cls_name in sorted(self.class_to_label.keys()):
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                logging.warning(f"[RenderedSST2] Class dir not found: {cls_dir}, skipping.")
                continue
            # 支持常见图像扩展�?            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                for p in cls_dir.glob(ext):
                    samples.append(p)

        if not samples:
            raise RuntimeError(f"[RenderedSST2] No images found under {split_dir}")

        self._samples = samples
        logging.info(f"[RenderedSST2] Loaded {len(samples)} images from {split_dir}")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if not self._samples:
            raise RuntimeError("[RenderedSST2] No samples loaded.")

        k = min(n, len(self._samples))
        idxs = np.random.choice(len(self._samples), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = self._samples[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[RenderedSST2] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[RenderedSST2] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(
            f"[RenderedSST2] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}"
        )
        return x
