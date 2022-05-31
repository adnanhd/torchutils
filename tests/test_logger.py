from torchutils.logging import PrintWriter
import torchutils

class DummyLogger(torchutils.logging.handler.TrainerLogger):
    def __init__(self):
        super(DummyLogger, self).__init__()

    def open(self, *args, **kwargs):
        ...

    def log(self, *args, **kwargs):
        ...

    def update(self, *args, **kwargs):
        ...

    def close(self, *args, **kwargs):
        ...

pw = PrintWriter()

def test_PrintWriter():
    pw.open(total=10)
    pw.log(loss=1e-2)
    pw.update()
    pw.close()
