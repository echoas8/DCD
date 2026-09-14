import scipy.io, torch, numpy as np, logging
from pathlib import Path
from PIL import Image
from .base import Task


class CarsTask(Task):
    key = "car"
    def load_inputs(self, preprocess, device, n: int):
        data_root = Path("./datasets/datasets/cars")
        annos = scipy.io.loadmat(data_root / "cars_annos.mat")
        imgs = []
        for a in annos["annotations"][0]:
            rel = Path(a[0][0].strip("/"))
            p = data_root / rel
            if p.exists(): imgs.append(p)
        choose = np.random.choice(len(imgs), size=min(n, len(imgs)), replace=True)
        tensors = [preprocess(Image.open(imgs[i]).convert("RGB")) for i in choose]
        x = torch.stack(tensors).to(device).requires_grad_(True)
        logging.info(f"[Cars] Loaded {len(choose)} samples, tensor {x.shape}")
        return x