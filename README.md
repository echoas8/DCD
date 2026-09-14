# Supplementary Code

This folder contains the supplementary code for Model-Specific Task Similarity for Vision–Language Model Selection via Layer Conductance

**Note:** The code has been anonymized for blind review. Some path dependencies may be incorrect and require adjustment before execution. However, the core logic for computing task representations in `frame_v2/` is correct and fully functional.

---

## 0. Environment

Tested on:
- OS: Ubuntu 22.04
- Python: 3.10
- CUDA: 12.8 
- GPU: A100 80GB*1

Create environment:
```bash
pip install -r requirements.txt
```

---

## 1. Datasets and Models

The datasets (21 tasks) and pre-trained models total approximately **160 GB** and cannot be included in the supplementary materials. Please download them separately:

- **Datasets:** Standard vision benchmarks (CIFAR-100, Cars, DTD, EuroSAT, etc.)
- **Models:** 48 CLIP-family models from OpenCLIP (automatically downloaded on first run)

---

## 2. Directory Structure

```
modelselection/
├── README.md
├── requirements.txt
├── frame_v2/                    # Core framework
│   ├── main.py                  # Entry point for conductance computation
│   ├── config.py                # Configuration dataclasses
│   ├── framework/
│   │   └── conductance.py       # Layer conductance computation
│   ├── models/
│   │   ├── clip_loader.py       # Model loading utilities
│   │   ├── extract_layers.py    # Visual encoder layer extraction
│   │   └── forward_funcs.py     # Forward functions for attribution
│   ├── tasks/                   # Task definitions (21 tasks)
│   │   ├── base.py
│   │   ├── cifar.py, cars.py, ...
│   └── viz/
│       └── plotting.py          # Visualization utilities
├── util/
│   ├── run_experiment.py        # Main experiment script
│   ├── run_experiment_symmetric.py  # Symmetric similarity ablation
│   ├── run_hyperparameter.py    # Hyperparameter
│   └── model-selection.xlsx     # Ground-truth
├── evaluation/                  # Zero-shot evaluation scripts
│   └── eva_*.py                 # Per-task evaluation (21 files)
├── datasets/                    # 21 tasks
│   └── datasets/
│       └── [task_name]/         # Images for each task
└── results/                     # Output directory
    └── result_{N}/              # Conductance vectors sampling from N image
```

---

## 3. Usage

### Compute Layer Conductance

```bash
python frame_v2/main.py --model ViT-B-32 --pretrained openai --n 100 --batch_size 2 --gpu 0
```

### Run Experiments

```bash
cd util
python run_experiment.py              # Main results
python run_experiment_symmetric.py    # Similarity ablation (Cosine / Soft-KL)
python run_hyperparameter.py          # Hyperparameter ablation (η, γ)
```

---

## 4. Key Hyperparameters

| Parameter | Description | 
|-----------|-------------|
| `N_tgt` | Number of unlabeled target images |
| `N_src` | Number of unlabeled images per source task | 
| `η` (eta) | Softmax temperature for target-conditioned block importance | 
| `γ` (gamma) | Softmin temperature for DCD-based source weighting | 

All are specified in `util/run_experiment.py`.

---