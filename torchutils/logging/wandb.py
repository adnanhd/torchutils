import wandb
import typing
import pandas as pd
import pdb
import warnings
import argparse

from .utils import Image
from .base import TrainerLogger

from ..models.utils import TrainerModel
from ..trainer.status import IterationStatus
from ..trainer.arguments import IterationArguments


class WandbLogger(TrainerLogger):
    __slots__ = ['_wandb', 'experiment', 'project', 'username', 'groupname']
    __wandb_logger__ = None

    def __init__(self,
                 experiment: str,
                 project: str,
                 username: str,
                 groupname: typing.Optional[str] = None,
                 finish_on_close: bool = True):
        wandb.login()
        self.experiment = experiment
        self.project = project
        self.username = username
        self.groupname = groupname
        self.finish_on_close = finish_on_close
        self._wandb: wandb.sdk.wandb_run.Run = None

    def open(self, hparams: IterationArguments):
        if self._wandb is None:
            self._wandb = wandb.init(project=self.project,
                                     entity=self.username,
                                     group=self.groupname,
                                     name=self.experiment,
                                     config=hparams.dict())
        else:
            warnings.warn(f"{self.__class__.__name__} is already opened. "
                          f"@{pdb.traceback.extract_stack()}", RuntimeWarning)

    def log_scores(self,
                   scores: typing.Dict[str, float],
                   status: IterationStatus):
        # @TODO: pass step parameter as an argument otherwise,
        # we don't know if it is current epoch or current step
        self._wandb.log(scores, step=status.current_epoch)

    def log_hparams(self,
                    params: argparse.Namespace,
                    status: IterationStatus):
        if not isinstance(params, dict):
            if hasattr(params, 'dict') and callable(params.dict):
                self._wandb.config.update(params.dict())
            else:
                self._wandb.config.update(params.__dict__)
        else:
            self._wandb.config.update(params)

    def log_table(self,
                  tables: typing.Dict[str, pd.DataFrame],
                  status: IterationStatus):
        to_log = dict()

        for table_name, data_frame in tables.items():
            if not isinstance(data_frame, pd.DataFrame):
                data_frame = pd.DataFrame(
                    columns=data_frame.keys(),
                    data=data_frame.values()
                )

            to_log[table_name] = wandb.Table(
                columns=data_frame.columns.to_list(),
                data=data_frame.values.tolist()
            )

        self._wandb.log(to_log, step=status.current_epoch)

    def log_image(self,
                  images: typing.Dict[str, Image],
                  status: IterationStatus):
        self._wandb.log({name: [wandb.Image(img) for img in image]
                        for name, image in images.items()})

    def log_module(self,
                   module: TrainerModel,
                   status: IterationStatus, **kwargs):
        self._wandb.watch(
            models=module.model,
            criterion=module.criterion,
            **kwargs
        )

    def update(self, n, status: IterationStatus):
        pass

    def close(self, status: IterationStatus):
        if self._wandb is not None:
            # @TODO: add quiet=True
            if self.finish_on_close:
                self._wandb.finish(exit_code=status.status_code)
            self._wandb = None
        else:
            warnings.warn(f"{self.__class__.__name__} is already closed. "
                          f"@{pdb.traceback.extract_stack()}", RuntimeWarning)
