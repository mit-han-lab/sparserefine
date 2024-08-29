from typing import Any, Dict, Optional

import torch
from torchmetrics import Metric

__all__ = ["Accuracy"]


class Accuracy(Metric):
    def __init__(
        self,
        output_key: str,
        target_key: str,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__(dist_sync_on_step=False)

        self.output_key = output_key
        self.target_key = target_key
        self.ignore_index = ignore_index

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, data: Dict[str, Any]) -> None:
        output = data[self.output_key]
        target = data[self.target_key]

        output = torch.argmax(output, dim=-1)

        if self.ignore_index is not None:
            output = output[target != self.ignore_index]
            target = target[target != self.ignore_index]

        self.tp += torch.sum(output == target)
        self.n += target.numel()

    def compute(self) -> torch.Tensor:
        return self.tp / self.n