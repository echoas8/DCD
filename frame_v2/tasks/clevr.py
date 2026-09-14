import os, logging, numpy as np, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from .base import Task


class _ClevrCountDataset(Dataset):
    def __init__(self, root, split="test", transform=None):
        self.root = os.path.join(str(root), split)
        self.transform = transform
        self.samples = []
        for fn in os.listdir(self.root):
            if not fn.endswith(".cls"): continue
            cls_path = os.path.join(self.root, fn)
            img_path = cls_path.replace(".cls", ".webp")
            if os.path.exists(img_path):
                self.samples.append((img_path,))
        logging.info(f"[CLEVR] {len(self.samples)} samples from {self.root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        (img_path,) = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return img


class ClevrTask(Task):
    key = "clevr"
    def load_inputs(self, preprocess, device, n: int):
        data_root = Path("./datasets/datasets/clevr_count")
        ds = _ClevrCountDataset(data_root, "test", transform=None)
        assert len(ds) > 0, "CLEVR dataset is empty or path is wrong."
        idxs = np.random.choice(len(ds), size=min(n, len(ds)), replace=True)
        tensors = [preprocess(ds[i]) for i in idxs]
        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[CLEVR] Loaded {len(idxs)} samples, tensor {x.shape}")
        return x