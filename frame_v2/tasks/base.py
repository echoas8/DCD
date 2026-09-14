from abc import ABC, abstractmethod
from torch import Tensor


class Task(ABC):
    key: str # registry key (e.g., "clevr")


    @abstractmethod
    def load_inputs(self, preprocess, device, n: int) -> Tensor:
        """Return a (N,C,H,W) tensor with requires_grad set."""
        raise NotImplementedError