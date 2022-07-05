import torch
import torchvision
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torchutils.data import Dataset
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.logging import ProgressBarLogger
from torchutils.callbacks import ModelCheckpoint, ScoreLoggerCallback

# Download ImageNet labels
dataset_length = 300000
dataset = Dataset(features=torch.randn(dataset_length, 512),
                  labels=torch.ones(dataset_length, 1))

model = TrainerModel(
    model=nn.Linear(512, 1),
    criterion='MSELoss',
    optimizer='SGD',
    lr=0.005,
    momentum=0.80,
)
trainer = Trainer(model=model, device='cuda')
trainer.compile_handlers(loggers=ProgressBarLogger.getLoggerGroup())
trainer.compile_handlers(callbacks=[ScoreLoggerCallback(), ModelCheckpoint()])
# trainer.compile(metrics=['loss'])
# trainer.compile(loggers=[pbar._step_bar])

trainer.train(num_epochs=4, batch_size=100,
              train_dataset=dataset, history={'Loss'})
