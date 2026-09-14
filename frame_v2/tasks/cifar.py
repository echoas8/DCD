import pickle, numpy as np, torch, logging
from pathlib import Path
from PIL import Image
from .base import Task


class CifarTask(Task):
    key = "cifar"
    def load_inputs(self, preprocess, device, n: int):
        data_root = Path("./datasets/datasets/cifar-100-python")
        meta_file = data_root / "meta"
        if not meta_file.exists():
            data_root = data_root / "cifar-100-python"
        with open(data_root / "test", "rb") as f:
            entry = pickle.load(f, encoding="bytes")
        arr = entry[b"data"]
        if arr.ndim == 2 and arr.shape[1] == 3072:
            images = arr.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        else:
            images = arr
        idxs = np.random.choice(len(images), size=min(n, len(images)), replace=True)
        tensors = [preprocess(Image.fromarray(images[i])) for i in idxs]
        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[CIFAR] Loaded {len(idxs)} samples, tensor {x.shape}")
        return x