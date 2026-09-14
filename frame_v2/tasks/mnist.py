# frame/tasks/mnist.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from .base import Task


class MNISTTask(Task):
    """
    MNIST 任务�?
    - 使用 torchvision.datasets.MNIST
    - 将灰度图转换�?RGB，再接入传入�?preprocess
    - 返回 (N, C, H, W) Tensor，requires_grad=True
    - label 在层导通率框架里不使用

    可通过环境变量覆盖�?        MNIST_ROOT  : MNIST 数据根目录（默认 ./data/mnist�?        MNIST_SPLIT : "train" �?"test"（默�?"test"�?    """
    key = "mnist"

    def __init__(self, root: str | None = None, split: str | None = None, download: bool | None = None):
        # 数据根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "MNIST_ROOT",
            root or "./data/mnist"
        ))

        # 使用 train 还是 test：环境变�?> 参数 > 默认 "test"
        self.split = os.environ.get("MNIST_SPLIT", split or "test").lower()
        self.train = self.split == "train"

        # 是否下载：默�?True（除非你手动关掉�?        env_download = os.environ.get("MNIST_DOWNLOAD", "")
        if download is not None:
            self.download = download
        elif env_download:
            self.download = env_download.lower() in ("1", "true", "yes", "y")
        else:
            self.download = True

        self._dataset = None  # 延迟构造，用到时再�?
    def _build_dataset(self, preprocess):
        # 灰度 -> RGB -> preprocess
        transform = transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB") if isinstance(im, Image.Image) else im),
            preprocess,
        ])
        self._dataset = datasets.MNIST(
            root=str(self.root),
            train=self.train,
            download=self.download,
            transform=transform,
        )
        logging.info(f"[MNIST] Loaded split='{self.split}' with {len(self._dataset)} samples from {self.root}")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if self._dataset is None:
            self._build_dataset(preprocess)

        if len(self._dataset) == 0:
            raise RuntimeError("[MNIST] Empty dataset.")

        k = min(n, len(self._dataset))
        idxs = np.random.choice(len(self._dataset), size=k, replace=True)

        tensors = []
        for i in idxs:
            img, _ = self._dataset[i]  # label 不需�?            tensors.append(img)

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[MNIST] Loaded {len(tensors)} samples, tensor {x.shape}")
        return x
