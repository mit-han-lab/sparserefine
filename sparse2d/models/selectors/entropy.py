from typing import Any, Dict

import torch
from torch import nn


__all__ = ["EntropySelector"]


class EntropySelector(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold


    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        probs = torch.softmax(inputs["logits"], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp(1e-5)), dim=-1)
        return entropy > self.threshold
