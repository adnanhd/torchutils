from typing import overload
from .handler import TrainerLogger
import os

class PrintWriter(TrainerLogger):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self):
        self(PrintWriter, self).__init__()

    @overload
    def log(self, *args, **kwargs):
        os.sys.stdout.write(*args, ", ".join(f'{key}={value}' for key, value in kwargs.items()))

    @overload
    def update(self, *args, **kwargs):
        os.sys.stdout.flush()
