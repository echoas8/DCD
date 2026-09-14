# frame/tasks/dtd.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class DTDTask(Task):
    """
    DTD (Describable Textures Dataset) 任务�?    读取 labels/test*.txt 里的图像相对路径，随机采�?n 张图片，
    �?preprocess 变成 (N,C,H,W) tensor 返回（用于层导通率）�?
    默认结构与原脚本一致：
        ROOT = "./datasets/datasets/dtd"
        图片目录: ROOT/dtd/images
        划分列表: ROOT/labels/test1.txt, test2.txt, test3.txt

    可通过环境变量覆盖�?        DTD_ROOT   : 数据根目�?        DTD_SPLITS : 列表文件名，逗号分隔，如 "test1.txt,test2.txt"
    """
    key = "dtd"

    def __init__(self, root: str | None = None, split_files: list[str] | None = None):
        # 1) 解析根目录（优先级：环境变量 > 参数 root > 默认�?        self.root = Path(os.environ.get(
            "DTD_ROOT",
            root or "./datasets/datasets/dtd"
        ))

        # 2) 解析划分列表文件�?        env_splits = os.environ.get("DTD_SPLITS", "")
        if split_files is not None:
            self.split_files = split_files
        elif env_splits:
            # 环境变量格式: "test1.txt,test2.txt"
            self.split_files = [s.strip() for s in env_splits.split(",") if s.strip()]
        else:
            # 默认使用 test1/2/3.txt
            self.split_files = [f"test{i}.txt" for i in range(1, 4)]

    # 内部：读取所有列表文件里的相对路径并拼成完整路径
    def _load_image_list(self) -> list[Path]:
        images_root = self.root / "dtd" / "images"
        labels_root = self.root / "labels"

        files: list[Path] = []
        for fname in self.split_files:
            list_path = labels_root / fname
            if not list_path.exists():
                logging.warning(f"[DTD] Split list not found: {list_path}, skipping.")
                continue
            with list_path.open("r") as f:
                for line in f:
                    rel = line.strip()
                    if not rel:
                        continue
                    files.append(images_root / rel)

        if not files:
            raise FileNotFoundError(
                f"[DTD] No images found. Checked splits: {self.split_files} under {labels_root}"
            )

        logging.info(f"[DTD] Collected {len(files)} image paths from {labels_root}")
        return files

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._load_image_list()
        assert len(files) > 0, "[DTD] Empty image list."

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[DTD] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[DTD] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[DTD] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
