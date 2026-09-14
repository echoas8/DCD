#!/usr/bin/env python3
# download_openclip.py

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download OpenCLIP weights into local cache.")
    parser.add_argument("--model", default="ViT-bigG-14", help='Model name, e.g. "ViT-bigG-14"')
    parser.add_argument("--pretrained", default="laion2b_s39b_b160k", help='Pretrained tag, e.g. "laion2b_s39b_b160k"')
    parser.add_argument("--cache-dir", default="", help="Cache dir for downloads (optional).")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"], help="Device to load model on.")
    args = parser.parse_args()

    # Optional: redirect common cache env vars so downloaded files land where you want.
    # open_clip uses huggingface_hub underneath for many weights, so HF_HOME helps.
    if args.cache_dir:
        cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        # Some setups also honor TORCH_HOME / XDG_CACHE_HOME
        os.environ["TORCH_HOME"] = cache_dir
        os.environ["XDG_CACHE_HOME"] = cache_dir
        print(f"[INFO] Using cache dir: {cache_dir}")

    try:
        import torch
        import open_clip
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install -U open_clip_torch torch\n"
            "If you want CUDA, install a CUDA-enabled torch build from PyTorch website."
        ) from e

    print(f"[INFO] Loading model={args.model}, pretrained={args.pretrained} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=args.device,
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    # Force a tiny forward to ensure everything is fully initialized.
    model.eval()
    with torch.no_grad():
        text = tokenizer(["hello"])
        _ = model.encode_text(text.to(args.device))

    print("[OK] Model and weights are available in your local cache now.")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] model class={type(model)}")
    print(f"[INFO] preprocess={preprocess}")

if __name__ == "__main__":
    main()
