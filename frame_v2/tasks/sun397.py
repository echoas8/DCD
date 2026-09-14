# frame/tasks/sun397.py
import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class SUN397Task(Task):
    key = "sun397"

    def __init__(self, root: str | None = None, split: str | None = None):
        # 原脚本里�?DATA_ROOT
        self.data_root = Path(os.environ.get(
            "SUN397_ROOT",
            root or "./datasets/datasets/sun397"
        ))
        # 图片�?DATA_ROOT/SUN397
        self.img_root = self.data_root / "SUN397"
        # 划分 json �?DATA_ROOT/split_zhou_SUN397.json
        self.split_json = self.data_root / "split_zhou_SUN397.json"

        self.split = os.environ.get("SUN397_SPLIT", split or "test")
        self._paths: list[Path] = []
        self._load_split()

    def _load_split(self):
        if not self.split_json.exists():
            raise FileNotFoundError(f"[SUN397] split json not found: {self.split_json}")

        with self.split_json.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if self.split not in data:
            raise KeyError(f"[SUN397] split '{self.split}' not in {list(data.keys())}")

        samples = data[self.split]
        paths: list[Path] = []
        for rel_path, _label, *_rest in samples:
            paths.append(self.img_root / rel_path)

        if not paths:
            raise RuntimeError(f"[SUN397] no samples for split '{self.split}'")

        self._paths = paths
        logging.info(f"[SUN397] Loaded split='{self.split}' with {len(self._paths)} images")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if not self._paths:
            raise RuntimeError("[SUN397] empty image list")

        k = min(n, len(self._paths))
        idxs = np.random.choice(len(self._paths), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = self._paths[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[SUN397] failed to open {p}: {e}; skip")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if not tensors:
            raise RuntimeError("[SUN397] all selected images failed to open")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[SUN397] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
