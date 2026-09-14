# frame/tasks/resisc45.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class RESISC45Task(Task):
    """
    RESISC45 任务
        DATA_ROOT = "./datasets/datasets/resisc45"
        txt 划分文件（例如）:
            test.txt / train800.txt / val200.txt / train800val200.txt
        每行格式�?            <relative/path> <label>
    """

    key = "resisc45"

    def __init__(self, root: str | None = None, split_txt: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "RESISC_ROOT",
            root or "./datasets/datasets/resisc45"
        ))
        # 划分文件名：环境变量 > 参数 > 默认 "test.txt"
        self.split_txt = os.environ.get("RESISC_SPLIT", split_txt or "test.txt")

    def _load_image_list(self) -> list[Path]:
        txt_path = self.root / self.split_txt
        if not txt_path.exists():
            raise FileNotFoundError(f"[RESISC45] Split file not found: {txt_path}")

        files: list[Path] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, _label = line.split()  # label 这里用不�?                img_path = self.root / rel_path
                files.append(img_path)

        if not files:
            raise RuntimeError(f"[RESISC45] No samples found in {txt_path}")

        logging.info(f"[RESISC45] Collected {len(files)} images from {txt_path}")
        return files

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._load_image_list()
        assert len(files) > 0, "[RESISC45] Empty image list."

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[RESISC45] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[RESISC45] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(
            f"[RESISC45] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}"
        )
        return x
