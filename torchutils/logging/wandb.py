from torchutils.logging import TrainerLogger
from torchutils.utils.pydantic import TrainerModel
from torch.nn import Module
from typing import Union
from collections import OrderedDict
import wandb

class WandBLogger(TrainerLogger):    
    __slots__ = ['_wandb']

    def open(self):
        #TODO: add config here
        self._wandb = wandb.init()

    def log_score(self, **kwargs):
        self._wandb.log(kwargs)

    def log_model(self, model: Union[TrainerModel, Module]):
        if isinstance(model, TrainerModel):
            model = model.model
        self._wandb.watch(model, log="all")

    def log_error(self, msg: str):
        pass

    def log_info(self, msg: str):
        pass
    
    def _flush_step(self):
        pass

    def _flush_epoch(self):
        pass

    def close(self):
        self._wandb = None

