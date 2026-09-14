# frame/tasks/kitti.py
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Task


class KITTITask(Task):
    """
    KITTI 任务（基�?closest_vehicle_distance txt 划分文件）：

    目录/文件约定与你原脚本一致：
        DATA_ROOT = "./datasets/datasets/kitti"
        图像路径�?txt 里是相对路径�?            <relative/path/to/image> <label_id>
        文本类别描述�?
            kitti_closest_vehicle_distance.txt   （这里只用图片，不用这个文件�?
    这里用于“层导通率”框架：
        - 从指�?txt 文件中读�?(相对路径, label) �?        - 将相对路径拼成完�?img_path
        - 随机采样 n 张，预处理后返回 (N, C, H, W) Tensor，requires_grad=True

    可通过环境变量覆盖默认配置�?        KITTI_ROOT  : 数据根目录（默认 ./datasets/datasets/kitti�?        KITTI_SPLIT : 使用的清单文件名（默�?"test.txt"，也可以�?"train800.txt" 等）
    """

    key = "kitti"

    def __init__(self, root: str | None = None, split_txt: str | None = None):
        # 根目录：环境变量 > 参数 > 默认
        self.root = Path(os.environ.get(
            "KITTI_ROOT",
            root or "./datasets/datasets/kitti"
        ))

        # 清单文件名：环境变量 > 参数 > 默认 "test.txt"
        self.split_txt = os.environ.get("KITTI_SPLIT", split_txt or "test.txt")

    def _load_image_list(self) -> list[Path]:
        """
        �?txt 文件中读取所有图像路径：
        每一行格式：<relative/path> <label>
        """
        txt_path = self.root / self.split_txt
        if not txt_path.exists():
            raise FileNotFoundError(f"[KITTI] Split file not found: {txt_path}")

        files: list[Path] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]  # 第二列是 label，这里用不到
                img_path = self.root / rel_path
                files.append(img_path)

        if not files:
            raise RuntimeError(f"[KITTI] No images found in {txt_path}")

        logging.info(f"[KITTI] Collected {len(files)} images from {txt_path}")
        return files

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        files = self._load_image_list()
        assert len(files) > 0, "[KITTI] Empty image list."

        k = min(n, len(files))
        idxs = np.random.choice(len(files), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = files[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[KITTI] Failed to open {p}: {e}; skipping.")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if len(tensors) == 0:
            raise RuntimeError("[KITTI] All selected images failed to open.")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[KITTI] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
