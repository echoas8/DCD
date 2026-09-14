import logging
import torch


def make_forward_func(model_name: str, model):
    name = model_name.lower()

    if name.startswith("rn"):
        def forward_func(x):
            out = model.visual(x)
            return torch.norm(out, p=2, dim=1)
        logging.info(f"Using simplified RN forward_func for {model_name}")
        return forward_func

    elif name.startswith("vit"):
        def forward_func(x):
            emb = model.encode_image(x)
            return torch.norm(emb, p=2, dim=1)
        logging.info(f"Using ViT forward_func for {model_name}")
        return forward_func

    elif name.startswith("convnext"):
        def forward_func(x):
            emb = model.encode_image(x)
            return torch.norm(emb, p=2, dim=1)
        logging.info(f"Using ConvNeXt forward_func for {model_name}")
        return forward_func

    elif name.startswith("eva"):
        def forward_func(x):
            emb = model.encode_image(x)
            return torch.norm(emb, p=2, dim=1)
        logging.info(f"Using EVA forward_func for {model_name}")
        return forward_func

    elif "siglip" in name:
        def forward_func(x):
            emb = model.encode_image(x)
            return torch.norm(emb, p=2, dim=1)
        logging.info(f"Using SigLIP ViT forward_func for {model_name}")
        return forward_func
    else:
        def forward_func(x):
            out = model.encode_image(x)
            return torch.norm(out, p=2, dim=1)
        logging.info(f"Using fallback forward_func for {model_name}")
        return forward_func
