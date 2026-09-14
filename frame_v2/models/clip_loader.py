import logging, open_clip, torch
from typing import Tuple
from frame_v2.models.forward_funcs import make_forward_func




def load_model(model_name: str, pretrained: str, device) -> Tuple[object, object, list, list, object]:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device).eval()
    from .extract_layers import extract_visual_layers # keep your util
    layers, layer_names = extract_visual_layers(model)
    logging.info(f"Model {model_name} loaded ({len(layers)} layers detected).")
    forward_func = make_forward_func(model_name, model)
    return model, preprocess, layers, layer_names, forward_func