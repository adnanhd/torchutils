# Copyright © 2021 Chris Hughes
import os
import copy
import torch
import typing
import math
from .callback import TrainerCallback
from ..utils import digest
from ..models import TrainerModel
import collections
import pydantic
from .utils import GoalEnum


class ModelCheckpoint(TrainerCallback):
    monitor: str
    goal: GoalEnum
    delta: float = pydantic.Field(0.0, validate_default=True)
    ckptpath: str = 'model.ckpt'
    init_from_checkpoint: bool = False
    halt_into_checkpoint: bool = False

    _model: TrainerModel = pydantic.PrivateAttr()
    _state_dict: collections.OrderedDict = pydantic.PrivateAttr()
    _score: float = pydantic.PrivateAttr()
    _delta: float = pydantic.PrivateAttr()
    _is_train: bool = pydantic.PrivateAttr()
    _history: list = pydantic.PrivateAttr()
    """
    Handles save and load operations between TrainerModel and Memory.

    Parameters
    ----------
        monitor : str
            name of the score to be kept track of
        trainer_model : TrainerModel
            the model of which weights are to be saved from and stored back
        save_path : str
            path on the memory of the TRAINER_MODEL's state_dict
        minimize_score : bool
            saves the TRAINER_MODEL's state dict when the monitored score is
            minimized with a margin of DELTA if True, or the score is maximized
            otherwise.
        delta : float
            a safety margin percentile of the current score from the saved
            score that makes the current checkpoint better than the saved one
        verbose : bool
            prints when checkpoint is saved, reset, and stored back the model
        load_back_per_epochs : int
            sets TRAINER_MODEL's state dict on the training end of every
            LOAD_BACK_PER_EPOCHS epochs. It is disabled when is zero.
        init_from_checkpoint : bool
            initializes the best checkpoint in the ModelCheckpoint with the one
            stored on SAVE_PATH on_initialization if True, or with None otwise.
        halt_into_checkpoint : bool
            stores the best checkpoint allocated in the ModelCheckpoint back to
            the SAVE_PATH on_termination if True, does noting otherwise.
        save_only_best_model : bool
            does not store the best checkpoint in the ModelCheckpoint back to
            the SAVE_PATH on_termination when the one on the same path have
            better score if True, does not check score parameter while storing
            otherwise.
        eval_with_best_model : bool
            loads the best checkpoint in the ModelCheckpoint to the TRAINER_MODEL
            on_evaluation_begin.
    """
    @pydantic.field_validator('delta')
    def validate_delta(cls, value, values: pydantic.ValidationInfo):
        return (1 + value) if values.data['goal'].value == 'maximize' else (1 - value)

    def __init__(self, monitor: str, model: TrainerModel, **kwds):
        super().__init__(monitor=monitor, readable_scores={monitor}, **kwds)
        # filesystemdaki weights
        # callabackteki checkpoint
        # modeldeki state_dict

        self._model: TrainerModel = model
        self._state_dict = model.state_dict()
        self._score: float = math.pow(-1, int(not self.maximize)) * math.inf

        # values to be stored
        self._history = list()
        # self.hparams = dict()

    @property
    def maximize(self):
        return self.goal.value == 'maximize'

    @property
    def current_run(self):
        return dict(#config=self.hparams,
                    scores=self._history,
                    monitor=self.monitor,
                    state_dict_checksum=digest(self._state_dict),
                    goal='maximize' if self.maximize else 'minimize')

    def _statedict_to_ckptfile(self):
        checkpoint = dict(state_dict=self._state_dict,
                          checksum=digest(self._state_dict),
                          runs=[self.current_run])
        if os.path.isfile(self.ckptpath):
            checkpoint['runs'].extend(torch.load(self.ckptpath)['runs'])
        torch.save(checkpoint, self.ckptpath)

    def _ckptfile_to_statedict(self):
        self._model.load_state_dict(
                torch.load(self.ckptpath)['state_dict'])

    def on_initialization(self):
        if self.init_from_checkpoint:
            if os.path.isfile(self.ckptpath):
                self._ckptfile_to_statedict()
            else:
                raise FileNotFoundError(f'ckptpath {self.ckptpath}')

    def on_termination(self):
        if self._is_train and self.halt_into_checkpoint:
            if os.path.isfile(self.ckptpath):
                run = torch.load(self.ckptpath)['runs'][0]
                score = run['scores'][-1][self.monitor]
                is_better = score * self.delta < self._score
            else:
                is_better = True

            if is_better:
                self._statedict_to_ckptfile()
                self.log_info(f"model saved into {self.ckptpath}")
            else:
                self.log_info("No better model saved into checkpoint")

    def on_training_begin(self, config: typing.Dict):
        #self.hparams = copy.deepcopy(config)
        self._is_train = True

    def on_evaluation_run_begin(self, config: typing.Dict):
        self._is_train = False

    def on_training_epoch_end(self):
        self._history.append({'status': 'epoch_end', **self.get_score_averages()})

    def on_validation_run_end(self):
        self._history.append({'status': 'valid_end', **self.get_score_averages()})
        score = self.get_score_averages()[self.monitor] if self.maximize else -self.get_score_averages()[self.monitor]

        if self._score * self.delta < score:
            self._score = score
            self._statedict_to_ckptfile()
