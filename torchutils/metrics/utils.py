import typing
import logging
import inspect
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class EndPoint:
    def __init__(self):
        for attr_name in dir(self):
            # if not initialized then continue
            if not hasattr(self, attr_name):
                continue

            attribute = getattr(self, attr_name)
            
            if isinstance(attribute, endpointmethod):
                for trigger in attribute.__triggers__:
                    def wrapped(attr):
                        def wrapper(*args, **kwds):
                            print(self._name)
                            print(self._sum, self._count, self.value)
                            attr(self, *args, **kwds)
                            print(self._sum, self._count, self.value)
                        return wrapper
                    
                    ## ??? ##
                    trigger.instance_callbacks[self].append(attribute)
                    setattr(self, attr_name, wrapped(attribute))


class endpointmethod:
    __slots__ = ['__triggers__', '__qualname__', '__func__']
    def __init__(self, func):
        print("INIT", self, func)
        self.__qualname__ = func.__qualname__
        self.__func__ = func
        self.__triggers__: typing.List[eventtrigger] = list()

    def __call__(self, *args, **kwds):
        self.__func__(*args, **kwds)


class eventtrigger:
    """ a decorator for creating a callback list """
    __slots__ = ['_name', 'callbacks', 'instances', 'instance_callbacks', '_parameters']

    def __init__(self, ep):
        self._name = ep.__qualname__
        self.callbacks = list()
        self.instances = list()
        self.instance_callbacks = defaultdict(list)
        self._parameters = inspect.signature(ep).parameters
        logger.debug(f'EventTriggerCreate: {ep.__qualname__} creates')

    def __call__(self, endpoint):
        if isinstance(endpoint, endpointmethod):
            endpoint.__triggers__.append(self)
            logger.debug(f'EventTriggerAssign: {self._name} assigns softly {endpoint.__qualname__}')
        else:
            self.callbacks.append(endpoint)
            logger.debug(f'EventTriggerAssign: {self._name} assigns {endpoint.__qualname__}')
        return endpoint

    def trigger(self, *args, **kwds):
        logger.debug(f'EventTriggerTriger: {self._name} trigger ({args} {kwds})')
        for callback in self.callbacks:
            callback(*args, **kwds)
        for instance, callbacks in self.instance_callbacks.items():
            for callback in callbacks:
                callback(instance, *args, **kwds)

    def __repr__(self):
        return f"<EventTrigger(@{self._name})>"
    


###############################################################################

"""
def SSS(*hookmethods: eventtrigger):
    #logger.debug(
    #    f'EventTriggerList{hookmethods}'
    #)

    class interfacemethod:
        pass

    class interface:
        "Decorator to create an interface to a hook.
        Requires a target hook as only argument. "
        __slots__ = ['eventname']

        def __init__(self, event: callable):
            self.eventname = event.__qualname__
            logger.debug(
                f'TRIGERLIST_CREATE: {event.__qualname__} will trigger {hookmethods}'
            )

        def __call__(self, fn):
            logger.debug(f'EVENT_TRIGGER: {self.eventname} -> {fn}')
            for hook in hookmethods:
                hook.append_callback(fn)
            return fn

        def method(self, mthd):
            logger.debug(f'{self.eventname}.method {mthd}')
            return mthd

    return interface


class autocall:
    pass
"""

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    class Handler:
        @eventtrigger
        def on_validation_end(self, foo: bool):
            pass

        @eventtrigger
        def on_training_step_end(self, bar: int):
            pass

        #@eventtrigger(on_training_step_end, on_validation_end)
        def OnStep(self):
            pass


    @Handler.on_training_step_end
    def function123(foo):
        logger.info(f'function123(foo: {foo})')

    Handler.on_training_step_end.trigger(123)

    class WandbLogger(EndPoint):
        def __init__(self, a):
            self.a = a
            super().__init__()

        @Handler.on_training_step_end
        @endpointmethod
        def log_scale(self, foo):
            print("LOG SCALE", foo, self.a)
    
    wdb_logger1 = WandbLogger(1)
    wdb_logger2 = WandbLogger(2)
    Handler.on_training_step_end.trigger(124)
