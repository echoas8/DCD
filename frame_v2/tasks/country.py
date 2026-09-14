import torch, numpy as np, logging
from pathlib import Path
from torchvision import datasets
from .base import Task

class CountryTask(Task):
    key = "country"
    def load_inputs(self, preprocess, device, n: int):
        root = Path("./datasets/datasets/country211/country211/test")
        ds = datasets.ImageFolder(root, transform=preprocess)
        idxs = np.random.choice(len(ds), size=min(n, len(ds)), replace=True)
        tensors = [ds[i][0] for i in idxs]
        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[Country211] Loaded {len(idxs)} samples, tensor {x.shape}")
        return x