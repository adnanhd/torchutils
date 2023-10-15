# Copyright © 2021 Chris Hughes
import os
import copy
import torch
import typing
import math
import warnings
import logging
from .callback import TrainerCallback
from ..models.hashs import digest
from ..models import TrainerModel


class ModelCheckpoint(TrainerCallback):
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

    def __init__(self,
                 model: TrainerModel,
                 monitor: str,
                 goal: str = 'minimize',
                 # patience: int = 0,
                 delta: float = 0.0,
                 verbose: bool = False,
                 filepath='model_checkpoint.ckpt',
                 init_from_checkpoint: bool = False,
                 halt_into_checkpoint: bool = False,
                 ):
        assert filepath.endswith('.ckpt')
        assert goal in ('minimize', 'maximize')
        super().__init__(level=logging.DEBUG if verbose else logging.INFO)
        # filesystemdaki weights
        # callabackteki checkpoint
        # modeldeki state_dict

        self.monitor: str = monitor
        self.model: TrainerModel = model
        self.state_dict = model.state_dict()
        self.delta: float = 1 + delta
        self.minimize: bool = goal == 'minimize'
        self.checkpoint_path: str = filepath
        self.score: float = math.pow(-1, goal == 'minimize') * math.inf

        # values to be stored
        self.history = list()
        self.hparams = dict()

        self.init_from_checkpoint: bool = init_from_checkpoint
        self.halt_into_checkpoint: bool = halt_into_checkpoint

    @property
    def current_run(self):
        return dict(config=self.hparams,
                    scores=self.history,
                    monitor=self.monitor,
                    state_dict_checksum=digest(self.state_dict),
                    goal='minimize' if self.minimize else 'maximize')

    def _statedict_to_ckptfile(self):
        checkpoint = dict(state_dict=self.state_dict,
                          checksum=digest(self.state_dict),
                          runs=[self.current_run])
        if os.path.isfile(self.checkpoint_path):
            checkpoint['runs'].extend(torch.load(self.checkpoint_path)['runs'])
        torch.save(checkpoint, self.checkpoint_path)

    def _ckptfile_to_statedict(self):
        self.model.load_state_dict(
                torch.load(self.checkpoint_path)['state_dict'])

    def on_initialization(self):
        if self.init_from_checkpoint:
            if os.path.isfile(self.checkpoint_path):
                self._ckptfile_to_statedict()
            else:
                raise FileNotFoundError(f'ckptpath {self.checkpoint_path}')

    def on_termination(self):
        if self.train and self.halt_into_checkpoint:
            if os.path.isfile(self.checkpoint_path):
                run = torch.load(self.checkpoint_path)['runs'][0]
                score = run['scores'][self.montior][0]
                is_better = score * self.delta < self.score
            else:
                is_better = True

            if is_better:
                self._statedict_to_ckptfile()
                self.logger.info(f"model saved into {self.checkpoint_path}")
            else:
                self.logger.info("No better model saved into checkpoint")

    def on_training_begin(self, config: typing.Dict):
        self.hparams = copy.deepcopy(config)
        self.train = True

    def on_evaluation_run_begin(self, config: typing.Dict):
        self.train = False

    def on_training_epoch_end(self):
        self.history.append({'status': 'epoch_end', **self.scores})

    def on_validation_run_end(self):
        self.history.append({'status': 'valid_end', **self.scores})
        score = math.pow(-1, self.minimize) * self.scores[self.monitor]

        if score * self.delta < self.score:
            self.score = score
            self._statedict_to_ckptfile()
