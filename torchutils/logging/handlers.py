import logging
try:
    import wandb
    wandb_imported = True
except ImportError:
    wandb_imported = False
from .filters import scoreFilter, scoreFilterRun
from .formatters import CsvFormatter


class CSVHandler(logging.FileHandler):
    # Pass the file name and header string to the constructor.
    def __init__(self, filename, *columns, mode='w', **kwds):
        assert isinstance(filename, str) and filename.endswith('csv')

        # Store the header information.
        # ('epoch', 'batch_index') + columns
        self.header = columns

        # Call the parent __init__
        super().__init__(filename=filename, mode=mode, **kwds)
        self.addFilter(scoreFilter)
        self.setFormatter(CsvFormatter(columns=self.header))


class WandbHandler(logging.Handler):
    def __init__(self, project, username, experiment_name, n=1, **kwds):
        assert wandb_imported, "install wandb package"
        self.wandb = wandb.init(project=project, username=username,
                                name=experiment_name, **kwds)
        super().__init__()
        self.addFilter(scoreFilter)

    def emit(self, record):
        if scoreFilterRun(record):
            metrics = {k + '/run': v for k, v in record.msg.items()}
        else:
            metrics = {k + '/step': v for k, v in record.msg.items()}
        self.wandb.log(metrics)

    def close(self):
        self.wandb.finish()
    pass
