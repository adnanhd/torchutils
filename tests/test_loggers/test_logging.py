import yaml
import logging.config
import logging
# import torchutils.logging.base as base


class TrainerLogger(object):
    __slots__ = ['config', 'logger']

    def __init__(self, fname, name=__name__):
        with open(fname, 'r') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        self.config = config
        self.logger = logging.getLogger(name=name)
        super(TrainerLogger, self).__init__()


class ScoreLogger(object):
    pass


class IterationInterface(object):
    __slots__ = ['handler']

    def __init__(self, handler: list):
        self.handler = handler
        super().__init__()

    @property
    def logger(self):
        return self.handler.logger


# logging_example.py

# logger = logging.getLogger(__name__)
logger = TrainerLogger('log/config.yaml')
interface = IterationInterface(handler=logger)


interface.logger.info('this is a info message')
interface.logger.debug('this is a debug message')
interface.logger.warning('This is a warning')
interface.logger.error('This is an error')
