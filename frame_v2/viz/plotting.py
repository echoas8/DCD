import matplotlib.pyplot as plt
from pathlib import Path


def visualize(results: dict, layer_names: list, output_dir: Path, model_name: str, pretrained: str):
    plt.figure(figsize=(10, 5))
    for task, scores in results.items():
        plt.plot(layer_names, scores, marker='o', linewidth=2, label=task)
    plt.title(f"Layer Conductance Comparison - {model_name}")
    plt.xlabel("Visual Layers / Blocks")
    plt.ylabel("Conductance (Raw Values)")
    plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout()
    png_path = output_dir / f"layer_conductance_{model_name.replace('-', '_').lower()}_{pretrained}.png"
    plt.savefig(png_path, dpi=300); plt.close()
    return png_path