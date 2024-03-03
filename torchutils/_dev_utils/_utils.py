import functools


def composite_function(*func):
    def compose(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose, func, lambda x: x)