# frame/tasks/eurosat.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class EuroSATTask(Task):
    """
    EuroSAT 任务�?    目录结构假定为（与你脚本中一致）�?        EUROSAT_ROOT = "./datasets/datasets/eurosat/2750"
        每个子目录是一个类别：EUROSAT_ROOT/<class_name>/*.jpg

    这里只做层导通率分析�?    - 遍历所有类别子目录，收集所有图像路�?    - 随机采样 n �?    - �?preprocess 预处理后拼成 (N, C, H, W) Tensor 返回
    """
    key = "eurosat"

    def __init__(self, root: str | None = None):
        # 优先级：环境变量 > 构造参�?> 默认硬编码路�?        self.root = Path(os.environ.get(
            "EUROSAT_ROOT",
            root or "./datasets/datasets/eurosat/2750"
        ))

    def _load_image_list(self) -> list[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"[EuroSAT] Root not found: {self.root}")

        # 类别子目�?        class_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda p: p.name)

        files: list[Path] = []
        for cdir in class_dirs:
            # 原脚本只用了 .jpg，这里也保持一�?            files.extend(sorted(cdir.glob("*.jpg")))

        if not files:
            raise RuntimeError(f"[EuroSAT] No .jpg images found under {self.root}")

        logging.info(f"[EuroSAT] Collected {len(files)} images from {len(class_dirs)} classes under {self.root}")
        return files

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._load_image_list()
        assert len(files) > 0, "[EuroSAT] Empty image list."

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[EuroSAT] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[EuroSAT] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[EuroSAT] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
