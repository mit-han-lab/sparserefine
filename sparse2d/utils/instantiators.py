import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from sparse2d.losses import LossCollection

from .modules import DataModule, ModelModule

__all__ = ["instantiate_data", "instantiate_model", "instantiate_trainer"]


def instantiate_data(cfg: DictConfig) -> L.LightningDataModule:
    return DataModule(instantiate(cfg.data.loaders))


def instantiate_model(cfg: DictConfig) -> L.LightningModule:
    model = instantiate(cfg.model)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    losses = LossCollection(**instantiate(cfg.losses))
    metrics = MetricCollection(dict(instantiate(cfg.metrics)), compute_groups=False)

    return ModelModule(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        losses=losses,
        metrics=metrics,
        cfg=cfg,
    )


def instantiate_trainer(cfg: DictConfig) -> L.Trainer:
    loggers = [instantiate(logger) for logger in cfg.loggers.values()]
    callbacks = [instantiate(callback) for callback in cfg.callbacks.values()]
    return L.Trainer(**instantiate(cfg.trainer), logger=loggers, callbacks=callbacks, detect_anomaly=True)
