import time
from torchutils.data import Dataset
import torch
from torch.nn import Linear
from torchutils.utils.pydantic import (
    TrainerStatus, TrainerModel, HandlerArguments
)

loader = Dataset(
    features=torch.randn(120, 42),
    labels=torch.randn(120, 1)
).dataloader(batch_size=30)


def __delay__(fn):
    def wrapped(*args, **kwargs):
        time.sleep(0.1)
        return fn(*args, **kwargs)
    return wrapped


class Pipeline():
    def __init__(self, epoch, batch, sample):
        self.args = HandlerArguments(
            model=TrainerModel(model=Linear(42, 1),
                               criterion='MSELoss',
                               optimizer='Adam',
                               lr=1e-2),
            status_ptr=[TrainerStatus()],
            train_dl=loader,
            valid_dl=loader,
            num_epochs=5,
        )
        self.epoch = epoch
        self.batch = batch
        self.sample = sample

    @__delay__
    def on_run_start(self):
        self.args.status.set_status_code(
            self.args.status.StatusCode.STARTED_SUCCESSFULLY
        )
        self.epoch.open(self.args)

    @__delay__
    def on_epoch_start(self):
        self.batch.open(self.args)

    @__delay__
    def on_step_end(self, idx):
        self.batch.log_scores({'loss': idx}, self.args.status)
        self.batch.update(1, self.args.status)

    @__delay__
    def on_valid_start(self):
        self.sample.open(self.args)

    @__delay__
    def on_valid_step_end(self, idx):
        self.sample.log_scores({'loss': idx}, self.args.status)
        self.sample.update(1, self.args.status)

    @__delay__
    def on_valid_end(self):
        self.sample.close(self.args.status)

    @__delay__
    def on_epoch_end(self, idx):
        self.batch.close(self.args.status)
        self.epoch.log_scores({'loss': idx}, self.args.status)
        self.epoch.update(1, self.args.status)

    @__delay__
    def on_run_end(self):
        self.args.status.set_status_code(
            self.args.status.StatusCode.FINISHED_SUCCESSFULLY
        )
        # self.epoch.update(1, self.args.status)
        self.epoch.close(self.args.status)

    def test_1(self):
        self.on_run_start()
        for epoch_idx in range(5):
            self.args.set_status(epoch=epoch_idx)
            self.on_epoch_start()
            for batch_idx in range(4):
                self.args.set_status(batch=batch_idx)
                self.on_step_end(batch_idx)
            if epoch_idx in [1, 3, 5]:
                self.on_valid_start()
                for batch_idx in range(4):
                    self.on_valid_step_end(batch_idx * batch_idx)
                self.on_valid_end()
            self.on_epoch_end(epoch_idx)
        self.on_run_end()
