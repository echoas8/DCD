# frame/tasks/patch_camelyon.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class PatchCamelyonTask(Task):
    """
    PatchCamelyon 任务�?        DATA_ROOT = "./datasets/datasets/patch_camelyon"
        TEST_TXT = os.path.join(DATA_ROOT, "test.txt")

        test.txt 每行�?            <relative/path> <label>

    """

    key = "patch_camelyon"

    def __init__(self, root: str | None = None, split_txt: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "PATCH_CAMELYON_ROOT",
            root or "./datasets/datasets/patch_camelyon"
        ))
        # txt 文件名：环境变量 > 参数 > 默认 "test.txt"
        self.split_txt = os.environ.get("PATCH_CAMELYON_SPLIT", split_txt or "test.txt")

    def _load_image_list(self) -> list[Path]:
        txt_path = self.root / self.split_txt
        if not txt_path.exists():
            raise FileNotFoundError(f"[PatchCamelyon] Split file not found: {txt_path}")

        files: list[Path] = []
        with txt_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, _label = line.split()  # label 不用
                full_path = self.root / rel_path
                files.append(full_path)

        if not files:
            raise RuntimeError(f"[PatchCamelyon] No samples found in {txt_path}")

        logging.info(f"[PatchCamelyon] Collected {len(files)} image paths from {txt_path}")
        return files

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._load_image_list()
        assert len(files) > 0, "[PatchCamelyon] Empty image list."

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[PatchCamelyon] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[PatchCamelyon] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(
            f"[PatchCamelyon] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}"
        )
        return x
