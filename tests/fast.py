import torch
import torchvision
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
# from torchutils.data import Dataset
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.datasets import NumpyDataset
from torchutils.datasets.transforms import ToDevice
import torchutils
#from torchutils.logging import ProgressBarLogger
#from torchutils.callbacks import ModelCheckpoint, ScoreLoggerCallback
from torchutils.metrics import MetricHandler

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
#trainer.train(num_epochs=4, batch_size=100)
# trainer = Trainer(model=model, device='cuda')
# trainer.compile_handlers(loggers=ProgressBarLogger.getLoggerGroup())
# trainer.compile_handlers(callbacks=[ScoreLoggerCallback(), ModelCheckpoint()])
# # trainer.compile(metrics=['loss'])
# # trainer.compile(loggers=[pbar._step_bar])
# 
# trainer.train(num_epochs=4, batch_size=100,
#               train_dataset=dataset, history={'Loss'})
import logging
from torchutils.trainer import CSVHandler
handler = CSVHandler("deneme.csv", "linear_model_loss")
etime = trainer.train(3, 100, handlers=[logging.FileHandler('results.txt'), handler])
# logging.basicConfig(level=0)
