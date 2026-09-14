import logging, numpy as np, torch
from captum.attr import LayerConductance
from typing import List




def compute_layer_conductance(model, layers: List[object], layer_names: List[str], forward_func, inputs, device, batch_size=1):
    scores = []
    model = model.to(device).eval()
    use_amp = torch.cuda.is_available()


    for name, layer in zip(layer_names, layers):
        lc = LayerConductance(forward_func, layer)
        try:
            batch_scores = []
            for batch in torch.split(inputs, max(1, len(inputs)//batch_size)):
                batch = batch.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    attr = lc.attribute(batch)
                batch_scores.append(attr.abs().mean().item())
                del attr
                torch.cuda.empty_cache()
            m = float(np.mean(batch_scores)) if batch_scores else 0.0
            scores.append(m)
            logging.info(f"Layer {name}: mean={m:.9f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning(f"⚠️ OOM on {name}, skipping layer.")
                torch.cuda.empty_cache()
                scores.append(0.0)
            else:
                raise
    return np.array(scores)