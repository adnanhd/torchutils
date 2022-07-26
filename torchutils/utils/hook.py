from abc import ABC, abstractmethod


def profile(fn):
    def wrapped_fn(*args, **kwargs):
        print(fn.__qualname__, *map(type, args), *
              map(lambda k, v: f'{k}={v}', kwargs.items()))
        return fn(*args, **kwargs)
    return wrapped_fn


"""
Caller is Event
Callee is Hooker

"""


class HookerClass:
    class Method:
        def __init__(self, fn):
            self.__call__ = fn

    # TODO: hooker class
    """Decorator to create a HookerClas."""

    @profile
    def __init__(self, func):
        pass

    @profile
    def __call__(self, *args, **kwargs):
        pass


class HookMethod:
    """Decorator to create a hook."""

    @profile
    def __init__(self, func):
        self._name = func.__name__
        self.callbacks = []

    @profile
    def __call__(self, *args, **kwargs):
        for callback in self.callbacks:
            print('callback is', callback, args, kwargs)
            callback(*args, **kwargs)

    @profile
    def __repr__(self):
        return f"<HookMethod(Hook={self._name})>"


def HookEvent(*hookMethods):
    """Decorator to create an interface to a hook.

    Requires a target hook as only argument.

    """
    @profile
    class HookInterface:
        __slots__ = ['_hookEvent']

        def __init__(self, hookEvent):
            self._hookEvent = hookEvent

        @profile
        def function(self, callback_fn):
            for hookMethod in hookMethods:
                hookMethod.callbacks.append(callback_fn)
            return callback_fn

        def method(self, callback_fn):
            return HookerClass.Method(callback_fn)

        def __call__(self, callback_fn):
            return self.function(callback_fn)

    return HookInterface
