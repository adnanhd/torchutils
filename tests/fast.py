import torch
import logging
import torch.nn as nn
from torchutils.logging import CSVHandler, scoreTrainStepFilter
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.datasets import NumpyDataset
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

trainer = Trainer(model=model, train_dataset=dataset)
handler_csv = CSVHandler("deneme.csv", "linear_model_loss")
handler_str = logging.StreamHandler()
handler_str.addFilter(scoreTrainStepFilter(50))
etime = trainer.train(3, 100, handlers=[handler_str])
