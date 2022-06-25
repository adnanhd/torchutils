# an acceptable error margin
epsilon = 1e-10


def isiterable(obj):
    if isinstance(obj, list):
        return True
    if isinstance(obj, tuple):
        return True
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def epsilon_equal(rhs, lhs):
    if isiterable(rhs) and isiterable(lhs):
        return all(epsilon_equal(l, r)
                   for r, l in zip(rhs, lhs))
    return abs(rhs - lhs) < epsilon
