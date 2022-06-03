from torchutils.logging import TrainerLogger
import wandb

class WBLogger(TrainerLogger):
    __slots__ = ['_name', '_wandb', '_config']
    def __init__(self, name):
        super(WBLogger, self).__init__()
        self._name = name
        self._config = dict()

    def log_model(self, model):
        self._wandb.watch(model, log="all")

    def config(self, **kwargs):
        self._config.update(kwargs)

    def open(self, *args, **kwargs):
        self._wandb = wandb.init(project=self._name, config=self._config)

    def log(self, **kwargs):
        self._wandb.log(kwargs)

    def close(self, *args, **kwargs):
        del self._wandb

    def update(self):
        ...

