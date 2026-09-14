# frame/tasks/oxford_flowers.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import scipy.io as sio

from .base import Task


class OxfordFlowersTask(Task):
    """
    Oxford Flowers 102 任务�?
    目录结构和文件命名与原脚本一致：
        DATA_ROOT = "./datasets/datasets/oxford_flowers"
        - 图像:    DATA_ROOT/jpg/image_00001.jpg ...
        - 标签:    DATA_ROOT/imagelabels.mat      (1..N, 每个�?1..102)
        - 划分:    DATA_ROOT/setid.mat            (trnid / valid / tstid)

    在层导通率框架中：
        - 根据 split 选择子集：train / val / test
        - 随机采样 n 张图�?        - 用传入的 preprocess 预处�?        - 返回 (N, C, H, W) Tensor，requires_grad=True

    可通过环境变量覆盖�?        OXF_FLOWERS_ROOT  : 数据根目�?        OXF_FLOWERS_SPLIT : "train" / "val" / "test" （默�?"test"�?    """

    key = "oxford_flowers"

    def __init__(self, root: str | None = None, split: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "OXF_FLOWERS_ROOT",
            root or "./datasets/datasets/oxford_flowers"
        ))

        # split：环境变�?> 参数 > 默认 "test"
        self.split = os.environ.get("OXF_FLOWERS_SPLIT", split or "test").lower()

        self.img_dir = self.root / "jpg"
        self._image_ids: np.ndarray | None = None  # e.g. array of ints

        self._load_split_ids()

    def _load_split_ids(self):
        labels_mat_path = self.root / "imagelabels.mat"
        setid_mat_path = self.root / "setid.mat"

        if not labels_mat_path.exists():
            raise FileNotFoundError(f"[OxfordFlowers] imagelabels.mat not found at {labels_mat_path}")
        if not setid_mat_path.exists():
            raise FileNotFoundError(f"[OxfordFlowers] setid.mat not found at {setid_mat_path}")

        labels_mat = sio.loadmat(labels_mat_path)
        setid_mat = sio.loadmat(setid_mat_path)

        # labels = labels_mat["labels"][0]  # 1..N, 这里不需要实际标签，只要 ids
        if self.split == "train":
            ids = setid_mat["trnid"][0]
        elif self.split == "val":
            ids = setid_mat["valid"][0]
        elif self.split == "test":
            ids = setid_mat["tstid"][0]
        else:
            raise ValueError(f"[OxfordFlowers] Unknown split: {self.split}, expected train/val/test")

        self._image_ids = ids
        logging.info(f"[OxfordFlowers] Loaded split='{self.split}' with {len(ids)} images from {self.root}")

    def _id_to_path(self, img_id: int) -> Path:
        # 按官方命名规�?image_00001.jpg
        return self.img_dir / f"image_{img_id:05d}.jpg"

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if self._image_ids is None or len(self._image_ids) == 0:
            raise RuntimeError("[OxfordFlowers] No image ids loaded.")

        k = min(n, len(self._image_ids))
        idxs = np.random.choice(len(self._image_ids), size=k, replace=True)

        tensors = []
        skipped = 0
        for idx in idxs:
            img_id = int(self._image_ids[idx])
            img_path = self._id_to_path(img_id)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logging.warning(f"[OxfordFlowers] Failed to open {img_path}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[OxfordFlowers] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[OxfordFlowers] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
