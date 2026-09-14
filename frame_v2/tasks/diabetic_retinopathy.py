# frame/tasks/diabetic_retinopathy.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class DiabeticRetinopathyTask(Task):
    """
    Diabetic Retinopathy (DR) 任务�?    从一�?txt 文件中读取图像相对路径和标签，这里只用路径，
    随机采样 n 张图片，经过 preprocess 后返�?(N,C,H,W) tensor�?    
    - 默认根目�? ./datasets/datasets/diabetic_retinopathy
    - 默认划分文件: test.txt
    - 可通过环境变量覆盖:
        DR_ROOT  : 数据根目�?        DR_SPLIT : 划分文件�?(�?test.txt / train.txt �?
    """
    key = "dr"

    def __init__(self, root: str | None = None, split_txt: str | None = None):
        # 允许通过环境变量覆盖默认路径
        self.root = Path(os.environ.get(
            "DR_ROOT",
            root or "./datasets/datasets/diabetic_retinopathy"
        ))
        self.split_txt = os.environ.get("DR_SPLIT", split_txt or "test.txt")

    def _load_image_list(self) -> list[Path]:
        txt_path = self.root / self.split_txt
        if not txt_path.exists():
            raise FileNotFoundError(f"[DR] Split file not found: {txt_path}")

        files: list[Path] = []
        with txt_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 行格式：<relative/path> <label>
                parts = line.split()
                rel_path = parts[0]
                p = self.root / rel_path  # test.txt 里相�?DATA_ROOT
                files.append(p)

        logging.info(f"[DR] Loaded {len(files)} paths from {txt_path}")
        return files

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._load_image_list()
        assert len(files) > 0, f"[DR] Empty list in {self.root}/{self.split_txt}"

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[DR] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[DR] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[DR] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
