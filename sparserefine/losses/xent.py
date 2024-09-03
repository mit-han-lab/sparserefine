from typing import Any, Dict

import torch
from torch import nn

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, output_key: str, target_key: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        output = data[self.output_key]
        target = data[self.target_key]
        return super().forward(output, target)