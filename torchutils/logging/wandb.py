from torchutils.trainer.utils import IterationArguments, IterationStatus
from torchutils.models.utils import TrainerModel
from torchutils.logging import TrainerLogger
from torch.nn import Module
import argparse
import wandb
import typing
import pandas as pd
from .utils import LoggingEvent
Image = typing.NewType(
    'Image', typing.Iterable[typing.Iterable[typing.Iterable[float]]]
)


class WandbLogger(TrainerLogger):
    __slots__ = ['_wandb', 'experiment', 'project', 'username', 'groupname']

    def __init__(self,
                 experiment: str,
                 project: str,
                 username: str,
                 groupname: typing.Optional[str] = None):
        wandb.login()
        self.experiment = experiment
        self.project = project
        self.username = username
        self.groupname = groupname
        self._wandb: wandb.sdk.wandb_run.Run = None

    def open(self, args: IterationArguments):
        self._wandb = wandb.init(project=self.project,
                                 entity=self.username,
                                 group=self.groupname,
                                 name=self.experiment,
                                 config=args.dict())

    @classmethod
    def getLogger(cls, event: LoggingEvent,
                  experiment: str,
                  project: str,
                  username: str,
                  groupname: str = None,
                  **kwargs) -> "TrainerLogger":
        if event == LoggingEvent.TRAINING_EPOCH:
            return cls(experiment=experiment, project=project,
                       username=username, groupname=groupname)

    def log_scores(self,
                   scores: typing.Dict[str, float],
                   status: IterationStatus):
        # @TODO: pass step parameter as an argument otherwise,
        # we don't know if it is current epoch or current step
        self._wandb.log(scores, step=status.current_epoch)

    def log_hyperparams(self,
                        params: argparse.Namespace,
                        status: IterationStatus):
        self._wandb.config.update(params.__dict__)

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

    def watch(self,
              module: typing.Union[Module, TrainerModel],
              status: IterationStatus, **kwargs):
        if isinstance(module, TrainerModel):
            module = module.module
            self._wandb.watch(
                models=module.module,
                criterion=module.criterion,
                **kwargs
            )
        else:
            self._wandb.watch(
                models=module,
                **kwargs
            )

    def update(self, n, status: IterationStatus):
        pass

    def close(self, status: IterationStatus):
        self._wandb.finish(quiet=True, exit_code=status.status_code)
        self._wandb = None
