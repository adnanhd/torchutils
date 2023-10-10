import logging
import csv
try:
    import wandb
    wandb_imported = True
except ImportError:
    wandb_imported = False
import io


TRAIN_STEP = 15
VALID_STEP = 15
EVAL_STEP = 15

TRAIN_EPOCH = 25
VALID_RUN = 25
EVAL_RUN = 25

logging.addLevelName(TRAIN_EPOCH, "TRAIN_EPOCH")
logging.addLevelName(TRAIN_STEP, "TRAIN_STEP")
logging.addLevelName(VALID_RUN, "VALID_RUN")
logging.addLevelName(VALID_STEP, "VALID_STEP")
logging.addLevelName(EVAL_RUN, "EVAL_RUN")
logging.addLevelName(EVAL_STEP, "EVAL_STEP")


def scoreFilter(record):
    return record.levelname in ("TRAIN_EPOCH", "TRAIN_STEP", "VALID_RUN", 
                                "VALID_STEP", "EVAL_RUN", "EVAL_STEP")


def scoreFilterRun(record):
    return record.levelname in ("TRAIN_EPOCH", "EVAL_RUN", "VALID_RUN")


def ScoreFilterStep(record):
    return record.levelname in ("VALID_STEP", "TRAIN_STEP", "EVAL_STEP")

       
# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CsvFormatter(logging.Formatter):
    def __init__(self, columns):
        super().__init__()
        self.header = columns
        self.written = False
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_NONNUMERIC)

    def _csv(self, rowlist) -> str:
        self.writer.writerow(rowlist)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

    def format(self, record):
        rowline = list(map(record.msg.__getitem__, self.header))
        if not self.written:
            self.written = True
            return "\n".join([self._csv(self.header), self._csv(rowline)])
        return self._csv(rowline)


class CSVHandler(logging.FileHandler):
    # Pass the file name and header string to the constructor.
    def __init__(self, filename, *columns, mode='w', **kwds):
        assert isinstance(filename, str) and filename.endswith('csv')

        # Store the header information.
        self.header = columns # ('epoch', 'batch_index') + columns

        # Call the parent __init__
        super().__init__(filename=filename, mode=mode, **kwds)
        self.addFilter(scoreFilter)
        self.setFormatter(CsvFormatter(columns=self.header))


class WandbHandler(logging.Handler):
    def __init__(self, project, entity, experiment_name):
        assert wandb_imported, "install wandb package"
        self.wandb = wandb.init(project=project, entity=entity, name=experiment_name)
        super().__init__()
        self.addFilter(scoreFilter)

    def emit(self, record):
        if scoreFilterRun(record):
            metrics = {key + '/run': value for key, value in record.msg.items()}
        else: # ScoreFilterStep
            metrics = {key + '/step': value for key, value in record.msg.items()}
        self.wandb.log(metrics)

    def close(self):
        self.wandb.finish()
    pass
