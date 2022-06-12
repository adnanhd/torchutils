## torchutils

### A Quict Example
```python
from torchutils.trainer import Trainer, TrainerModel
from torchutils.data import Dataset
import torch.nn as nn
import numpy as np


class Model(nn.Module):
	def __init__(sel):
		...
	def forward(self, X):
		...
		return X

model = TrainerModel(model= Model(), loss=nn.MSELoss())
trainer = Trainer(model=model, device='cuda')
train_dataset = Dataset(features=..., labels=...)
valid_dataset = train_dataset.split(0.2)
triner.compile(metrics=['loss'])
run_history = trainer.train(num_epochs=120, 
						    learning_rate=1e-2, 
			  			    batch_size=25,
              			    train_dataset=train_dataset,
              			    valid_dataset=valid_dataset,
              			    num_epochs_per_validation=10)
```
