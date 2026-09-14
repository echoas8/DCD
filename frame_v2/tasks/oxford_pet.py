# frame/tasks/oxford_pet.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class OxfordPetTask(Task):
    """
    Oxford-IIIT Pet 任务�?
    目录结构和原脚本一致：
        DATA_ROOT = "./datasets/datasets/oxford-pet"
        - 图片:      DATA_ROOT/images/*.jpg
        - 标注 & 划分:
            DATA_ROOT/annotations/trainval.txt
            DATA_ROOT/annotations/test.txt

        trainval.txt / test.txt 每行格式�?            <img_name> <class_id> <species> <breed_id>
            例如：Abyssinian_1 1 0 0

    在层导通率框架中：
        - 根据 split (train/test) 选择对应 txt
        - 读取 img_name，构�?images/<img_name>.jpg
        - 随机采样 n 张图片，preprocess 后返�?(N, C, H, W) Tensor，requires_grad=True

    可通过环境变量覆盖�?        OXFPET_ROOT  : 数据根目录（默认上面的路径）
        OXFPET_SPLIT : "train" �?"test"（默�?"test"�?    """

    key = "oxford_pet"

    def __init__(self, root: str | None = None, split: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "OXFPET_ROOT",
            root or "./datasets/datasets/oxford-pet"
        ))

        # split：环境变�?> 参数 > 默认 "test"
        self.split = os.environ.get("OXFPET_SPLIT", split or "test").lower()

        self.img_dir = self.root / "images"
        ann_dir = self.root / "annotations"
        self.split_file = ann_dir / ("trainval.txt" if self.split == "train" else "test.txt")

        self._samples: list[tuple[str, int]] = []
        self._load_split()

    def _load_split(self):
        if not self.split_file.exists():
            raise FileNotFoundError(f"[OxfordPet] Split file not found: {self.split_file}")

        samples: list[tuple[str, int]] = []
        with self.split_file.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, class_id_str = parts[0], parts[1]
                    try:
                        class_id = int(class_id_str) - 1  # �?0-based，但这里其实不用 label
                    except ValueError:
                        class_id = 0
                    samples.append((img_name, class_id))

        if not samples:
            raise RuntimeError(f"[OxfordPet] No samples found in {self.split_file}")

        self._samples = samples
        logging.info(f"[OxfordPet] Loaded split='{self.split}' with {len(samples)} samples from {self.split_file}")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if not self._samples:
            raise RuntimeError("[OxfordPet] No samples loaded.")

        k = min(n, len(self._samples))
        idxs = np.random.choice(len(self._samples), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            img_name, _ = self._samples[i]
            img_path = self.img_dir / f"{img_name}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logging.warning(f"[OxfordPet] Failed to open {img_path}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[OxfordPet] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[OxfordPet] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
