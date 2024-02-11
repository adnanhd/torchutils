import torch
import logging
import torch.nn as nn
from torchutils.logging import CSVHandler, scoreFilterRun
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.datasets.datasets import TensorDataset
from torchutils.callbacks import LogMetersCallback
import sys
import typing
from torchutils._dev_utils import MetricType, AverageMeter

# Download ImageNet labels
dataset_length = 300000
model = TrainerModel(
    model=nn.Linear(512, 1).to(device='cuda'),
    criterion='mse_loss',
    optimizer='SGD',
    scheduler='ReduceLROnPlateau',
    lr=0.005,
    momentum=0.80,
)


@MetricType.register
def regression_metrics(batch_output: torch.Tensor,
                       batch_target: torch.Tensor,
                       **batch_extra_kwds) -> typing.Dict[str, float]:
    return {'l1_loss': torch.nn.functional.l1_loss(batch_output, batch_target).item()}
AverageMeter(name='l1_loss')


dataset = TensorDataset(torch.randn(dataset_length, 512),
                        torch.ones(dataset_length, 1))

dataset.map(lambda sample: tuple(map(torch.Tensor.cuda, sample)))

trainer = Trainer(model=model, log_level=10)
#handler_csv = CSVHandler("deneme.csv", "linear_model_loss")
handler_str = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler_str.setFormatter(formatter)
logger = LogMetersCallback(model.criterion_name, 'l1_loss', level=10)
#handler_str.addFilter(scoreFilterRun(50))
# logging.basicConfig(level=10)
etime = trainer.train(3, 100, handlers=[handler_str],
                      callbacks=[logger], train_dataset=dataset,
                      num_epochs_per_validation=2,
                      metrics={'regression_metrics'})
