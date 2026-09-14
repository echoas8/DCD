from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ModelCfg:
    name: str = "RN50"
    pretrained: str = "openai"


@dataclass
class RunCfg:
    seed: int = 42
    gpu_id: str = "1"
    n_per_task: int = 2
    batch_size: int = 2
    output_dir: Path = Path("results")
    milestones: List[int] = field(default_factory=list)


@dataclass
class AppCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    run: RunCfg = field(default_factory=RunCfg)
    tasks: List[str] = field(default_factory=lambda: ["car", "cifar", "country", "clevr", "dmlab", "dr", "dtd", "eurosat"]) # task registry keys