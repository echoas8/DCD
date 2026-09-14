# frame/tasks/gtsrb.py
import os
import csv
import glob
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class GTSRBTask(Task):
    """
    GTSRB 任务�?    目录结构假定为（与你原脚本一致）�?
        GTSRB_ROOT = "./datasets/datasets/gtsrb"

        训练�?        GTSRB_ROOT/GTSRB/Training/<class_id>/*.ppm|*.png|*.jpg

        测试�?        GTSRB_ROOT/GTSRB/Final_Test/Images/*.ppm|*.png|*.jpg
        GTSRB_ROOT/GT-final_test.csv   # �?Filename, ClassId, ...

    在层导通率框架中：
        - 根据 split（train/test）收集所有图片路�?        - 随机采样 n �?        - 使用传入�?preprocess 预处�?        - 返回 (N, C, H, W) �?Tensor（requires_grad=True�?
    环境变量覆盖�?        GTSRB_ROOT  : 数据根目�?        GTSRB_SPLIT : "train" �?"test"（默�?"test"�?    """
    key = "gtsrb"

    def __init__(self, root: str | None = None, split: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "GTSRB_ROOT",
            root or "./datasets/datasets/gtsrb"
        ))

        # split：环境变�?> 参数 > 默认 "test"
        self.split = os.environ.get("GTSRB_SPLIT", split or "test").lower()

        # 路径定义（与原脚本一致）
        self.train_dir = self.root / "GTSRB" / "Training"
        self.test_dir = self.root / "GTSRB" / "Final_Test" / "Images"
        self.gt_csv = self.root / "GT-final_test.csv"

    def _load_train_files(self) -> list[Path]:
        if not self.train_dir.exists():
            raise FileNotFoundError(f"[GTSRB] Training dir not found: {self.train_dir}")

        files: list[Path] = []
        for class_dir in sorted(os.listdir(self.train_dir)):
            class_path = self.train_dir / class_dir
            if not class_path.is_dir():
                continue
            # 扫描多种扩展�?            for ext in ("*.ppm", "*.png", "*.jpg", "*.jpeg"):
                files.extend(Path(class_path).glob(ext))

        if not files:
            raise RuntimeError(f"[GTSRB] No training images found under {self.train_dir}")
        logging.info(f"[GTSRB] Collected {len(files)} training images.")
        return files

    def _load_test_files(self) -> list[Path]:
        if not self.gt_csv.exists():
            raise FileNotFoundError(f"[GTSRB] Missing GT-final_test.csv at {self.gt_csv}")
        if not self.test_dir.exists():
            raise FileNotFoundError(f"[GTSRB] Test images dir not found: {self.test_dir}")

        files: list[Path] = []
        with self.gt_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_path = self.test_dir / row["Filename"]
                if img_path.is_file():
                    files.append(img_path)

        if not files:
            raise RuntimeError(f"[GTSRB] No test images found using {self.gt_csv}")
        logging.info(f"[GTSRB] Collected {len(files)} test images.")
        return files

    def _collect_files(self) -> list[Path]:
        if self.split == "train":
            return self._load_train_files()
        elif self.split == "test":
            return self._load_test_files()
        else:
            raise ValueError(f"[GTSRB] Unknown split: {self.split} (expected 'train' or 'test')")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._collect_files()
        assert len(files) > 0, "[GTSRB] Empty image list."

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[GTSRB] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[GTSRB] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[GTSRB] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
