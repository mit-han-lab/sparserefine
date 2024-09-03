from typing import Any, Dict, Callable

import lightning as L
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MetricCollection
import torch
from sparserefine.losses import LossCollection
import torch.distributed as dist

__all__ = ["ModelModule"]


class ModelModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        losses: LossCollection,
        metrics: MetricCollection,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.model = model
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.losses = losses
        self.metrics = metrics
        self.save_hyperparameters(cfg, ignore=["model", "optimizer", "scheduler", "losses", "metrics"])

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(data)

    def training_step(self, data: Dict[str, Any], batch_idx) -> Dict[str, Any]:
        
        return self._shared_step(data, split="train", on_step=True)

    def validation_step(self, data: Dict[str, Any], batch_idx) -> Dict[str, Any]:
        return self._shared_step(data, split="val", on_step=False)

    def test_step(self, data: Dict[str, Any], batch_idx) -> Dict[str, Any]:
        return self._shared_step(data, split="val", on_step=False)

    def _shared_step(self, data: Dict[str, Any], split: str, on_step: bool = False) -> Dict[str, Any]:
        data = self(data)
            
        loss, losses = self.losses(data)
        if split == "val":
            self.metrics.update(data)

        if self.trainer is not None:
            self.log(
                f"{split}/loss",
                loss.detach(),
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )
            self.log_dict(
                {f"{split}/losses/{key}": val.detach() for key, val in losses.items()},
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )

        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def _log_epoch_metrics(self, split: str) -> None:
        metrics = self.metrics.compute()
        for key, val in metrics.items():
            self.log(f"{split}/metrics/{key}", val, sync_dist=True)
        self.metrics.reset()

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    
    
    