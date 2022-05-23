from typing import List, Dict, Any, Mapping, Optional, Union, Callable, Tuple, Iterable
from abc import ABC
from ...callbacks import TrainerCallback
from ...callbacks.hooks import HookMethod


class MetricHandler(TrainerCallback, ABC):
    verbose = False

    def call_event(self, event, *args, **kwargs):
        hook = self.__getattribute__(event)
        if self.verbose: print(hook._name, "calls", *hook.callbacks)
        hook(*args, **kwargs)


def callback_function():
    pass


for fn_name, fn in vars(TrainerCallback).items():
    if 'on_' in fn_name:
        setattr(MetricHandler, fn_name, HookMethod(fn))
