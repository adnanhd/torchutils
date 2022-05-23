import time

verbose = False

def profile(fn):
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


