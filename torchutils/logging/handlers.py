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
        self.wandb = wandb.init(project=project, entity=username,
                                name=experiment_name, **kwds)
        super().__init__()
        self.addFilter(scoreFilter)

    def emit(self, record):
        if record.levelname == 'VALID_STEP':
            metrics = {k + '/valid_step': v for k, v in record.msg.items()}
        elif record.levelname == 'VALID_RUN':
            metrics = {k + '/valid_run': v for k, v in record.msg.items()}
        elif record.levelname == 'TRAIN_STEP':
            metrics = {k + '/train_step': v for k, v in record.msg.items()}
        elif record.levelname == 'TRAIN_EPOCH':
            metrics = {k + '/train_epoch': v for k, v in record.msg.items()}
        else:
            metrics = record.msg
        self.wandb.log(metrics)

    def close(self):
        self.wandb.finish()
