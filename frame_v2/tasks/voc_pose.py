# frame/tasks/voc_pose.py
import os
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image

from .base import Task


class VOCPoseTask(Task):
    key = "voc_pose"

    def __init__(self, root: str | None = None):
        # 原脚�?DATA_ROOT = ".../VOCdevkit/VOC2007"
        self.root = Path(os.environ.get(
            "VOC_ROOT",
            root or "./datasets/datasets/voc2007/VOCdevkit/VOC2007"
        ))
        self.jpeg_dir = self.root / "JPEGImages"
        self.annot_dir = self.root / "Annotations"
        self._paths: list[Path] = []
        self._load_from_xml()

    def _load_from_xml(self):
        if not self.annot_dir.exists():
            raise FileNotFoundError(f"[VOC] Annotations dir not found: {self.annot_dir}")
        if not self.jpeg_dir.exists():
            raise FileNotFoundError(f"[VOC] JPEGImages dir not found: {self.jpeg_dir}")

        xml_files = [f for f in os.listdir(self.annot_dir) if f.endswith(".xml")]
        paths: list[Path] = []

        for xml_file in xml_files:
            xml_path = self.annot_dir / xml_file
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                filename_node = root.find("filename")
                if filename_node is None or not filename_node.text:
                    continue
                filename = filename_node.text
                img_path = self.jpeg_dir / filename
                if img_path.exists():
                    paths.append(img_path)
            except Exception as e:
                logging.warning(f"[VOC] failed to parse {xml_path}: {e}; skip")

        if not paths:
            raise RuntimeError(f"[VOC] no images found from {self.annot_dir}")

        self._paths = paths
        logging.info(f"[VOC] Loaded {len(self._paths)} images from annotations")

    def load_inputs(self, preprocess, device, n: int) -> torch.Tensor:
        if not self._paths:
            raise RuntimeError("[VOC] empty image list")

        k = min(n, len(self._paths))
        idxs = np.random.choice(len(self._paths), size=k, replace=True)

        tensors = []
        skipped = 0
        for i in idxs:
            p = self._paths[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logging.warning(f"[VOC] failed to open {p}: {e}; skip")
                skipped += 1
                continue
            tensors.append(preprocess(img))

        if not tensors:
            raise RuntimeError("[VOC] all selected images failed to open")

        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[VOC] Loaded {len(tensors)} samples (skipped {skipped}), tensor {x.shape}")
        return x
