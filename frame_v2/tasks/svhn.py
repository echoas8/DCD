# frame/tasks/svhn.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class SVHNTask(Task):
    key = "svhn"

    def __init__(self, root: str | None = None, split_txt: str | None = None):
        self.root = Path(os.environ.get(
            "SVHN_ROOT",
            root or "./datasets/datasets/svhn"
        ))
        # 原脚本是用各�?txt，例�?test.txt / train800.txt
        self.split_txt = os.environ.get("SVHN_SPLIT", split_txt or "test.txt")
        self._paths: list[Path] = []
        self._load_list()

    def _load_list(self):
        txt_path = self.root / self.split_txt
        if not txt_path.exists():
            raise FileNotFoundError(f"[SVHN] split file not found: {txt_path}")

        paths: list[Path] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, _label = line.split()
                img_path = self.root / rel_path
                paths.append(img_path)

        if not paths:
            raise RuntimeError(f"[SVHN] no samples found in {txt_path}")

        self._paths = paths
        logging.info(f"[SVHN] Loaded {len(self._paths)} images from {txt_path}")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if not self._paths:
            raise RuntimeError("[SVHN] empty image list")

        k = min(n, len(self._paths))
        idxs = np.random.choice(len(self._paths), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = self._paths[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[SVHN] failed to open {p}: {e}; skip")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if not tensors:
            raise RuntimeError("[SVHN] all selected images failed to open")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[SVHN] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
