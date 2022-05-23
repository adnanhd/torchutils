from abc import ABC, abstractmethod


class HookerClass:
    # TODO: hooker class
    """Decorator to create a HookerClas."""

    def __init__(self, func):
        pass

    def __call__(self, *args, **kwargs):
        pass


class HookMethod:
    """Decorator to create a hook."""

    def __init__(self, func):
        self._name = func.__name__
        self.callbacks = []

    def __call__(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)

    def __repr__(self):
        return f"<HookMethod(Hook={self._name})>"


def HookEvent(*hookMethods):
    """Decorator to create an interface to a hook.

    Requires a target hook as only argument.

    """
    def HookInterface(hookEvent):
        def add_callback(callback_fn):
            for hookMethod in hookMethods:
                hookMethod.callbacks.append(callback_fn)
            return callback_fn
        return add_callback
    return HookInterface
