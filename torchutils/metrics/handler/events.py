from ...callbacks.hooks import HookEvent
from .handler import MetricHandler

@HookEvent(MetricHandler.on_initialization)
def on_initialization(func): pass

@HookEvent(MetricHandler.on_training_epoch_begin, 
           MetricHandler.on_validation_run_begin,
           MetricHandler.on_evaluation_run_begin)
def on_run_begin(func): pass

@HookEvent(MetricHandler.on_training_epoch_end, 
           MetricHandler.on_validation_run_end,
           MetricHandler.on_evaluation_run_end)
def on_run_end(func): pass

@HookEvent(MetricHandler.on_training_step_begin, 
           MetricHandler.on_validation_step_begin,
           MetricHandler.on_evaluation_step_begin)
def on_step_begin(func): pass

@HookEvent(MetricHandler.on_training_step_end, 
           MetricHandler.on_validation_step_end,
           MetricHandler.on_evaluation_step_end)
def on_step_end(func): pass
