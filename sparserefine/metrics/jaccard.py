from typing import Any, Dict, Optional

import torch
from torchmetrics import Metric

__all__ = ["JaccardIndex"]


class JaccardIndex(Metric):
    def __init__(
        self,
        output_key: str,
        target_key: str,
        num_classes: int,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__(dist_sync_on_step=False)

        self.output_key = output_key
        self.target_key = target_key
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.add_state("i", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
        self.add_state("u", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")

    def update(self, data: Dict[str, Any]) -> None:
        output = data[self.output_key]
        target = data[self.target_key]

        output = torch.argmax(output, dim=-1)

        if self.ignore_index is not None:
            output = output[target != self.ignore_index]
            target = target[target != self.ignore_index]

        for c in range(self.num_classes):
            self.i[c] += torch.sum((output == c) & (target == c))
            self.u[c] += torch.sum((output == c) | (target == c))

    def compute(self) -> torch.Tensor:
        return torch.mean(self.i / self.u.clamp(min=1e-5))