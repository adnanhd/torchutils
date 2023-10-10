import typing
import logging
import inspect

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def get_non_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is inspect.Parameter.empty
    }


class hookermethod:
    """ a decorator for creating a callback list """
    __slots__ = ['_name', 'callbacks', '_parameters']

    def __init__(self, func):
        self._name = func.__name__
        self.callbacks = list()
        self._parameters = inspect.signature(func).parameters
        logger.debug(f'{self.__class__.__qualname__}.__init__ '
                     f'{func.__qualname__} parameters: '
                     f'{inspect.signature(func).parameters}')

    def append_callback(self, callback):
        logger.debug(f'{self.__call__.__qualname__}.append_callback'
                     f'{callback.__qualname__} with parameters'
                     f'{inspect.signature(callback).parameters}')
        self.callbacks.append(callback)

    def __call__(self, *args, **kwargs):
        logger.debug(
            f'{self.__class__.__qualname__}.__call__ is called {args} {kwargs}')
        for callback in self.callbacks:
            logger.debug(f'callback is {callback}')
            callback(*args, **kwargs)

    def __repr__(self):
        return f"<HookMethod(Hook={self._name})>"


def eventmethod(*hookmethods: hookermethod):
    logger.debug(
        f'{eventmethod.__qualname__} {hookmethods}'
    )

    class interfacemethod:
        pass

    class interface:
        """Decorator to create an interface to a hook.

        Requires a target hook as only argument.

        """
        __slots__ = []

        def __init__(self, event: callable):
            logger.debug(
                f'{self.__class__.__qualname__}.__init__ {event.__qualname__}'
            )

        def __call__(self, fn):
            return self.function(fn)

        def function(self, fn):
            logger.debug(f'{self.__class__.__qualname__}.function {fn}')
            for hook in hookmethods:
                hook.append_callback(fn)
            return fn

        def method(self, mthd):
            logger.debug(f'{self.__class__.__qualname__}.method {fn}')
            return interfacemethod(mthd)

    return interface


class autocall:
    pass

###############################################################################


class Handler:
    @hookermethod
    def on_validation_end(self, foo: bool):
        pass

    @hookermethod
    def on_training_step_end(self, bar: int):
        pass

    @eventmethod(on_training_step_end, on_validation_end)
    def OnStep(self):
        pass


@Handler.OnStep
def function(foo):
    logger.info('i am called')


class WandbLogger:
    @Handler.OnStep.function
    def log_scale(self, foo, bar):
        pass

# class Handler(object):
#     class OnStepEnd():
#         @hookmethod
#         def log_score(self):
#             pass
#
#     class OnValidStep:
#         @hookmethod
#         def log_score(self):
#             pass
#
#
# @Handler.OnStepEnd.log_score
# @Handler.OnValidStep.log_score
# def OnStepEnd():
#     pass
#
#
# class WandbLogger():
#     @OnStepEnd
#     def log_score(self):
#         pass
