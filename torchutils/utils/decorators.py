import os, time, logging
#create a logger
logger = logging.getLogger(__name__)
#set logging level
logger.setLevel(logging.DEBUG)

# create a log file
try:
    os.mkdir('log')
except FileExistsError:
    pass
handler = logging.FileHandler(f'log/{__name__}.log')

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def verbose(fn):
    def wrapped_fn(*args, **kwargs):
        logger.debug(f'{fn.__name__} <-- ' + ' '.join(map(str, args)) + ' ' + ' '.join(f'{k}={v}' for k, v in kwargs.items()))
        res = fn(*args, **kwargs)
        logger.debug(f'{fn.__name__} --> {res.__repr__()}')
        return res
    return wrapped_fn

def profile(fn):
    def wrapped_fn(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        finish = time.time()
        dTime = '%.3e' % (finish - start)
        logger.debug(f'{fn.__name__}@{dTime}')
        return res
    return wrapped_fn

