from .utils import LoggingEvent
from .base import TrainerLogger
from .proxy import LoggerProxy
from .handler import LoggerHandler
from .tty import SlurmLogger
from .nop import NoneLogger

import torchutils.logging.tty
import torchutils.logging.pbar

import importlib
if importlib.util.find_spec('tqdm') is not None:
    from .pbar import ProgressBarLogger

if importlib.util.find_spec('wandb') is not None:
    from .wandb import WandbLogger as WandBLogger
del importlib
