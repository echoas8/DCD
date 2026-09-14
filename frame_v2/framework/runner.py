import json, logging, numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Type


from frame_v2.config import AppCfg
from frame_v2.models.clip_loader import load_model
from frame_v2.framework.conductance import compute_layer_conductance


# --- simple registry ---
_TASKS: Dict[str, Type[object]] = {}


def register(task_cls):
    _TASKS[task_cls.key] = task_cls
    return task_cls


def run(cfg: AppCfg):
    device = None
    from ..utils.env import setup_environment
    device = setup_environment(cfg.run.seed, cfg.run.gpu_id)


    model, preprocess, layers, layer_names, forward_func = load_model(
        cfg.model.name, cfg.model.pretrained, device
    )

    # instantiate tasks
    tasks = []
    for key in cfg.tasks:
        if key not in _TASKS:
            raise KeyError(f"Task '{key}' not registered.")
        tasks.append(_TASKS[key]())

    outdir = cfg.run.output_dir
    
    # Determine milestones
    total_n = cfg.run.n_per_task
    ms = sorted(list(set(cfg.run.milestones + [total_n])))
    
    # Filter milestones > total_n? Or just respect them? 
    # User said: "until 50", assuming 50 is total. 
    # If user provides milestones [20, 50] and n=150, we should probably process 20, 50, 150.
    # The set(...) handles duplicates.
    
    # Store running states: key -> {"n": int, "scores": np.array}
    task_states = {}
    
    final_results = {} 

    for m in ms:
        if m > total_n:
            logging.warning(f"Milestone {m} > n_per_task {total_n}, capping at {total_n}")
            m = total_n
            
        current_outdir = outdir / f"result_{m}"
        current_outdir.mkdir(parents=True, exist_ok=True)
        
        step_results = {}
        
        logging.info(f"=== Reaching milestone N={m} ===")

        for t in tasks:
            state = task_states.get(t.key, {"n": 0, "scores": None})
            prev_n = state["n"]
            prev_scores = state["scores"]
            
            delta_n = m - prev_n
            
            if delta_n <= 0:
                step_results[t.key] = prev_scores
                continue

            logging.info(f"\n🔍 Processing task: {t.key} (delta={delta_n})")
            inputs = t.load_inputs(preprocess, device, delta_n)
            
            # Compute scores for the new batch
            new_scores = compute_layer_conductance(
                model, layers, layer_names, forward_func, inputs, device, cfg.run.batch_size
            )
            
            # Update running average
            if prev_scores is None:
                updated_scores = new_scores
            else:
                # Weighted average: (prev * prev_n + new * delta_n) / m
                updated_scores = (prev_scores * prev_n + new_scores * delta_n) / m
            
            task_states[t.key] = {"n": m, "scores": updated_scores}
            step_results[t.key] = updated_scores
            logging.info(f"✅ {t.key} updated (N={m}).")

        # Save results for this milestone
        np.save(current_outdir / f"layer_conductance_{cfg.model.name.replace('-', '_').lower()}_{cfg.model.pretrained}.npy", step_results)
        meta = {"cfg": asdict(cfg), "layer_names": layer_names, "milestone": m}
        (current_outdir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        logging.info(f"✅ Results saved for milestone {m}.")
        
        final_results = step_results # Keep the last one as return value

    return final_results, layer_names