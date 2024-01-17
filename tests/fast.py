import torch
import logging
import torch.nn as nn
from torchutils.logging import CSVHandler, scoreFilterRun
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.datasets import NumpyDataset
from torchutils.callbacks import AverageScoreLogger
import sys
from torchutils.datasets.transforms import ToDevice

# Download ImageNet labels
dataset_length = 300000
model = TrainerModel(
    model=nn.Linear(512, 1).to(device='cuda'),
    criterion='MSELoss',
    optimizer='SGD',
    scheduler='ReduceLROnPlateau',
    lr=0.005,
    momentum=0.80,
)

dataset = NumpyDataset(inputs=torch.randn(dataset_length, 512),
                       labels=torch.ones(dataset_length, 1),
                       input_transforms=[ToDevice('cuda')],
                       label_transforms=[ToDevice('cuda')])

trainer = Trainer(model=model, train_dataset=dataset, log_level=10)
handler_csv = CSVHandler("deneme.csv", "linear_model_loss")
handler_str = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler_str.setFormatter(formatter)
logger = AverageScoreLogger(model.criterion_name, level=20)
#handler_str.addFilter(scoreFilterRun(50))
# logging.basicConfig(level=10)
etime = trainer.train(3, 100, handlers=[handler_str],
                      callbacks=[logger],
                      num_epochs_per_validation=2)
