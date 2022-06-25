from torchutils.utils.pydantic import TrainerModel, HandlerArguments, TrainerStatus
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
        self._wandb = None

    def open(self, args: HandlerArguments):
        self._wandb = wandb.init(
            project=self.project,
            entity=self.username,
            group=self.groupname,
            name=self.experiment,
            config=args.hparams.dict()
        )

    @classmethod
    def getLogger(cls, event: LoggingEvent,
                  experiment: str = None,
                  project: str = None,
                  username: str = None,
                  groupname: str = None,
                  **kwargs) -> "TrainerLogger":
        return cls(experiment=experiment, project=project,
                   username=username, groupname=groupname)

    def log_scores(self,
                   scores: typing.Dict[str, float],
                   status: TrainerStatus):
        self._wandb.log(scores)

    def log_hyperparams(self,
                        params: argparse.Namespace,
                        status: TrainerStatus):
        self._wandb.config.update(params.__dict__)

    def log_table(self,
                  tables: typing.Dict[str, pd.DataFrame],
                  status: TrainerStatus):
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
                  status: TrainerStatus):
        self._wandb.log({name: [wandb.Image(img) for img in image]
                        for name, image in images.items()})

    def watch(self,
              module: typing.Union[Module, TrainerModel],
              status: TrainerStatus, **kwargs):
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

    def update(self, n, status: TrainerStatus):
        pass

    def close(self, status: TrainerStatus):
        self._wandb.finish(quiet=True, exit_code=status.status_code)
        self._wandb = None
