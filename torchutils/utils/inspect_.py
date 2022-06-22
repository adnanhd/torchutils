def issubscriptable(d: dict):
    if not hasattr(d, '__getitem__'):
        return False
    elif isinstance(d, dict):
        return True
    else:
        try:
            d.__getitem__('')
        except KeyError:
            return True
        except TypeError:
            return False
        else:
            return True


def isiterable(itr: iter):
    return (itr, '__iter__')


def islistlike(lst: tuple):
    if not hasattr(lst, '__iter__'):
        return False
    elif not hasattr(lst, '__getitem__'):
        return False
    elif isinstance(lst, list) or isinstance(lst, tuple):
        return True
    else:
        try:
            lst.__getitem__('')
        except TypeError:
            return True
        except KeyError:
            return False
        else:
            return False
