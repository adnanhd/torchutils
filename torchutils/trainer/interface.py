import pydantic
import torch
import typing
from ..metrics.handler import MetricHandler
from ..metrics.history import RunHistory
from .batch import IterationBatch
from .status import IterationStatus
from .arguments import IterationArguments


class IterationInterface(pydantic.BaseModel):
    batch: IterationBatch = IterationBatch()
    status: IterationStatus = IterationStatus()
    hparams: IterationArguments = pydantic.Field(None)

    _at_epoch_end: bool = pydantic.PrivateAttr(False)
    _metric_handler: MetricHandler = pydantic.PrivateAttr()
    _metric_history: RunHistory = pydantic.PrivateAttr()
    _score_names: typing.Set[str] = pydantic.PrivateAttr()

    def __init__(self,
                 metrics: MetricHandler,
                 history: RunHistory,
                 hparams: IterationArguments):
        super().__init__()
        self.hparams = hparams
        self._metric_handler = metrics
        self._metric_history = history
        score_names = self._metric_history.get_score_names()
        self._score_names = score_names

    # functions for IterationHandler
    def collate_fn(self, input, preds, target):
        self._at_epoch_end = False
        self.batch.collate_fn(input, preds, target)
        self._metric_handler.run_score_functional(preds=preds,
                                                  target=target)

    def set_metric_scores(self):
        self._at_epoch_end = True

    def reset_metric_scores(self):
        score_values = self._metric_handler.get_score_averages(
            *self._metric_history.get_score_names())
        for score_name, score_value in score_values.items():
            self._metric_history.set_latest_score(score_name, score_value)
        # @TODO: instead of (postincrementing) _increment_epoch
        # use allocate_score_values (preincrementing)
        self._metric_history._increment_epoch()
        self._metric_handler.reset_score_values()

    # functions for Callbacks and Third Party Comp.

    def get_current_scores(self,
                           *score_names: str
                           ) -> typing.Dict[str, float]:
        """ Returns the latest step or epoch values, depending on
        whether it has finished itereting over the current epoch or not """
        # @TODO: make calls without star
        if self._at_epoch_end:
            return self._metric_handler.get_score_averages(*score_names)
        else:
            return self._metric_handler.get_score_values(*score_names)

    def get_stored_scores(
            self,
            *score_names: str
    ) -> typing.Dict[str, typing.List[float]]:
        """ Returns the all epoch values with given score names """
        return {
            score_name: self._metric_history.get_score_values(score_name)
            for score_name in score_names
        }
