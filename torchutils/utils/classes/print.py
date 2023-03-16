class PrintOnce(object):
    def __init__(self):
        self._printed = False

    def __call__(self, *args, **kwargs):
        if not self._printed:
            print(*args, **kwargs)
            self._printed = True
