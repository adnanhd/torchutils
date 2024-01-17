import torch
import logging
import torch.nn as nn
from torchutils.logging import CSVHandler, scoreFilterStep, formatter
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.datasets import NumpyDataset
from torchutils.datasets.transforms import ToDevice
from torchutils.callbacks import ModelCheckpoint

# Download ImageNet labels
dataset_length = 30000
model = TrainerModel(
    model=nn.Linear(512, 1).to(device='cuda'),
    criterion='MSELoss',
    optimizer='SGD',
    scheduler='ReduceLROnPlateau',
    modelname='Linear Model',
    lr=0.005,
    momentum=0.80,
)

dataset = NumpyDataset(inputs=torch.randn(dataset_length, 512),
                       labels=torch.ones(dataset_length, 1),
                       input_transforms=[ToDevice('cuda')],
                       label_transforms=[ToDevice('cuda')])

trainer = Trainer(model=model, train_dataset=dataset, valid_dataset=dataset)
handler_csv = CSVHandler("deneme.csv", "linear_model_loss")
handler_str = logging.StreamHandler()
handler_str.addFilter(scoreFilterStep(6))
handler_str.setFormatter(formatter)
etime = trainer.train(num_epochs=5, batch_size=1000,
                      handlers=[handler_str], metrics={'linear_model_loss'},
                      callbacks=[ModelCheckpoint(monitor=model.criterion_name, goal='minimize', filepath='model.ckpt', model=model, halt_into_checkpoint=True)])
