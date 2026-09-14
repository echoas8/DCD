# frame/tasks/fer2013.py
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .base import Task


class FER2013Task(Task):
    """
    FER2013 任务�?    - 数据来自一�?CSV（默�?train.csv�?    - CSV 中包含列�?        - 'pixels' : 空格分隔�?48*48 灰度�?        - 'emotion': 可选（这里不使用，只做图像特征分析�?        - 'Usage'  : train / publictest / privatetest �?    
    这里仅用于层导通率�?    - 根据 split 选择子集（默�?'train'�?    - 随机采样 n 张，�?48x48 转为 RGB PIL，再�?preprocess 处理
    - 返回 (N, C, H, W) �?Tensor（requires_grad=True�?
    可通过环境变量覆盖�?        FER_ROOT  : 数据根目录（默认 /.../fer2013�?        FER_CSV   : CSV 文件名（默认 'train.csv'�?        FER_SPLIT : 使用�?split 关键字（默认 'train'�?    """
    key = "fer2013"

    def __init__(
        self,
        root: str | None = None,
        csv_name: str | None = None,
        split: str | None = None,
    ):
        # 1) 根目录：环境变量 > 参数 > 默认路径
        self.root = Path(os.environ.get(
            "FER_ROOT",
            root or "./datasets/datasets/fer2013"
        ))

        # 2) CSV 文件�?        self.csv_name = os.environ.get("FER_CSV", csv_name or "train.csv")

        # 3) split 关键字（作用�?Usage 列）
        self.split = os.environ.get("FER_SPLIT", split or "train").lower()

        self._images: list[str] = []
        self._load_csv()

    def _load_csv(self):
        csv_path = self.root / self.csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"[FER2013] CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if "Usage" in df.columns:
            df = df[df["Usage"].str.lower().str.contains(self.split)]
            logging.info(f"[FER2013] Filtered split='{self.split}', remaining {len(df)} rows.")
        else:
            logging.warning("[FER2013] Column 'Usage' not found, using all rows.")

        if "pixels" not in df.columns:
            raise KeyError("[FER2013] CSV must contain a 'pixels' column.")

        self._images = df["pixels"].astype(str).tolist()
        logging.info(f"[FER2013] Loaded {len(self._images)} images from {csv_path}")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if not self._images:
            raise RuntimeError("[FER2013] No images loaded from CSV.")

        k = min(n, len(self._images))
        idxs = np.random.choice(len(self._images), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            pixels_str = self._images[i]
            try:
                # '48*48' 灰度 -> numpy -> PIL -> RGB
                arr = np.array(pixels_str.split(), dtype=np.uint8).reshape(48, 48)
                img = Image.fromarray(arr).convert("RGB")
            except Exception as e:
                logging.warning(f"[FER2013] Failed to decode row {i}: {e}; skipping.")
                skipped += 1
                continue

            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[FER2013] All selected images failed to decode.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[FER2013] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
