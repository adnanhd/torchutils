from .handler import TrainerLogger


class NoneLogger(TrainerLogger):
    def __init__(self):
        super(NoneLogger, self).__init__()

    def open(self, *args, **kwargs):
        ...

    def log(self, *args, **kwargs):
        ...

    def update(self, *args, **kwargs):
        ...

    def close(self, *args, **kwargs):
        ...

