from torchutils.metrics import TrainerMetric, register_to_metric_handler


@register_to_metric_handler
class LossTracker(TrainerMetric):
    def __init__(self):
        self._loss = None
        super().__init__()

    def set_scores(self, x, y, y_pred):
        pass

    @property
    def score_names(self) -> set:
        return {'loss'}

    @property
    def loss(self) -> float:
        return self._loss
