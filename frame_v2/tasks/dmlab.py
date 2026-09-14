# frame/tasks/dmlab.py
import os
import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from .base import Task


class DMLabTask(Task):
    """
    读取 DMLab �?split 清单文件（如 test.txt），按行解析出图像相对路径与标签�?    这里只取图像路径，随机采�?n 张并返回预处理后�?Tensor（requires_grad=True）�?    - 数据根目录可通过环境变量 DMLAB_ROOT 覆盖
    - 划分文件名可通过环境变量 DMLAB_SPLIT 覆盖
    """
    key = "dmlab"

    def __init__(self, root: str | None = None, split_txt: str | None = None):
        self.root = Path(os.environ.get(
            "DMLAB_ROOT",
            root or "./datasets/datasets/dmlab"
        ))
        self.split_txt = os.environ.get("DMLAB_SPLIT", split_txt or "test.txt")

    def _load_image_list(self) -> list[Path]:
        txt_path = self.root / self.split_txt
        if not txt_path.exists():
            raise FileNotFoundError(f"[DMLab] Split file not found: {txt_path}")

        files: list[Path] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 行格式：<relative/path/to/image> <label>
                parts = line.split()
                rel = parts[0]
                p = Path(rel)
                if not p.is_absolute():
                    p = self.root / rel  # DMLab 通常是相对路�?                files.append(p)
        return files

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._load_image_list()
        assert len(files) > 0, f"[DMLab] Empty list in {self.root}/{self.split_txt}"

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[DMLab] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[DMLab] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[DMLab] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
