from .utils import loss_to_metric, one_hot_decode
from .base import TrainerMetric, MetricHandler

def register_to_metric_handler(trainer_metric_class):
    MetricHandler.register_metric(trainer_metric_class())
    return trainer_metric_class
