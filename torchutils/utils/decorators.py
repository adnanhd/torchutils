import os, time, logging
#create a logger
logger = logging.getLogger(__name__)
#set logging level
logger.setLevel(logging.DEBUG)


def log_to_file(path: str):
    global logger
    # create a log file
    dirname, basename = os.path.split(path)
    os.makedirs(dirname, exist_ok=True)
    handler = logging.FileHandler(path)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # add a logging destination (file)
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

verbose = False
def profile2(fn):
    if verbose:
        def wrapped_function(*args, **kwargs):
            now = time.time()
            print('>>', fn.__name__, *args, ", ".join(f'{k}={v}' for k,v in kwargs.items()))
            res = fn(*args, **kwargs)
            print('<<', fn.__name__, 'takes', "%.3e" % (time.time() - now))
            return res
        return wrapped_function
    else:
        def wrapped_function(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapped_function
