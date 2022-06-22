from typing import Union, Optional, Dict, Any, List
from torchutils.utils.pydantic import TrainerModel
from torchutils.logging import TrainerLogger
from torch.nn import Module
import argparse
import wandb
from .utils import LoggingEvent


class WandbLogger(TrainerLogger):
    __slots__ = ['_wandb']

    def open(self):
        # TODO: add config here
        self._wandb = wandb.init()

    @classmethod
    def getLogger(cls, event: LoggingEvent) -> TrainerLogger:
        return cls()

    def log_scores(self,
                   scores: Dict[str, float],
                   step: Optional[int] = None):
        self._wandb.log(scores)

    def log_hyperparams(self,
                        params: argparse.Namespace,
                        step: Optional[int] = None):
        self._wandb.log(params.__dict__)

    def log_table(self,
                  key: str,
                  table: Dict[str, List[Any]]):
        self._wandb.log({key: wandb.Table(columns=table.keys(),
                                          data=table.values())})

    def log_image(self,
                  key: str,
                  images: List[Any],
                  step: Optional[int] = None):
        self._wandb.log({key: [wandb.Image(image) for image in images]})

    def watch(self, module: Union[Module, TrainerModel], log='all', **kwargs):
        if isinstance(module, TrainerModel):
            module = module.module
        self._wandb.watch(module, log=log, **kwargs)

    def close(self):
        self._wandb = None
