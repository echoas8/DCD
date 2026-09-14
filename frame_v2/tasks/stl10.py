# frame/tasks/stl10.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class STL10Task(Task):
    """
    STL10 任务（binary 版本）：
        DATA_ROOT = "./datasets/datasets/stl10/stl10_binary"

        - train_X.bin / train_y.bin
        - test_X.bin  / test_y.bin
        - class_names.txt

    """

    key = "stl10"

    def __init__(self, root: str | None = None, split: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "STL10_ROOT",
            root or "./datasets/datasets/stl10/stl10_binary"
        ))

        # split：环境变�?> 参数 > 默认 "test"
        self.split = os.environ.get("STL10_SPLIT", split or "test").lower()
        assert self.split in ("train", "test"), f"[STL10] Unsupported split: {self.split}"

        # 对应的二进制文件�?        if self.split == "train":
            self.data_file = self.root / "train_X.bin"
            self.label_file = self.root / "train_y.bin"
        else:
            self.data_file = self.root / "test_X.bin"
            self.label_file = self.root / "test_y.bin"

        self._data: np.ndarray | None = None  # (N, H, W, C)
        self._load_binary()

    def _load_binary(self):
        if not self.data_file.exists():
            raise FileNotFoundError(f"[STL10] data file not found: {self.data_file}")

        # 读取图像数据
        with self.data_file.open("rb") as f:
            arr = np.fromfile(f, dtype=np.uint8)

        # 每张图像 3x96x96
        try:
            arr = arr.reshape(-1, 3, 96, 96)
        except ValueError as e:
            raise RuntimeError(f"[STL10] Failed to reshape binary data: {e}")

        # 转成 HWC 方便 PIL.fromarray
        arr = np.transpose(arr, (0, 2, 3, 1))
        self._data = arr
        logging.info(f"[STL10] Loaded split='{self.split}' with {len(self._data)} images from {self.data_file}")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if self._data is None or len(self._data) == 0:
            raise RuntimeError("[STL10] No data loaded.")

        k = min(n, len(self._data))
        idxs = np.random.choice(len(self._data), size=k, replace=True)

        tensors = []
        for i in idxs:
            img_arr = self._data[i]  # HWC, uint8
            img = Image.fromarray(img_arr)
            tensors.append(preprocess(img))

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[STL10] Loaded {len(tensors)} samples, tensor {x.shape}")
        return x
