from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["DiceLoss"]


class DiceLoss(nn.Module):
    def __init__(
        self,
        output_key: str,
        target_key: str,
        ignore_index: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        output = data[self.output_key]
        target = data[self.target_key]

        if self.ignore_index is not None:
            output = output[target != self.ignore_index]
            target = target[target != self.ignore_index]

        output = F.softmax(output, dim=-1)
        target = torch.eye(output.shape[-1], dtype=output.dtype, device=output.device)[target.long()]

        intersection = torch.sum(output * target, dim=0)
        cardinality = torch.sum(output + target, dim=0)
        return 1 - (2 * intersection / (cardinality + self.eps)).mean()