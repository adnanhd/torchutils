## torchutils
PyTorch-Utils is a library to boost the use of PyTorch library using only very few lines of codes.

### A Quick Example
```python
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
import torch.nn as nn


class Model(nn.Module):
  def __init__(sel):
    ...
  def forward(self, X):
    ...
    return X

class MyTrainerModel(TrainerModel):
  def forward(self, batch, batch_index=None):
    ...

model = MyTrainerModel(
  model=Model(), 
  criterion='MSELoss', 
  optimizer='SDG',
  scheduler='ReduceLROnPleateau',
  lr=1e-5, momentum=0.9,
)

train_dataset = ...
valid_dataset = ...

trainer = Trainer(model=model, train_dataset=train_dataset, valid_dataset=valid_dataset)
trainer.train(num_epochs=120, batch_size=24)
```
