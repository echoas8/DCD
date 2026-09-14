import os, random, logging, torch, numpy as np


def setup_environment(seed: int = 42, gpu_id: str = "0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.info(f"Using device: {device}, seed: {seed}")
    return device