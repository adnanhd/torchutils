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
  model=Model(), ## equivalenlty model='model-name' if 'model-name' is registerd in torchvision.models
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

## TO-DO
### Important
- [ ] Add torch.distributed support
- [x] `Metric Handler.compute()` functions 
- [x] and metric registration mechanism
  - [ ] adding some regression, classification, segmentation metrics
  - [ ] support also jit or ray computation
- [x] passing `batch` and `batch_output` to `MetricHandler.compute()` only, not to `CallbackHandler`



### Less-Important
- [ ] Add configs for BuilderTypes
- [ ] profilers logging etc.

### Trivial
- [x] get rid of DataLoaderWrapper instead use datasets.DatasetWrapper
- [ ] register some well known datasets like mnist, cifar-10, cifar-100, voc

- [ ] fix AverageScore registration, overwrite if it's exists and fix `.add_score()` and `.add_score_name()` functions

- [ ] Update `ScoreLoggerCallback` for Logging Levels: set only `SCORE` as a level, or set two `SCORE`s where step=15, epoch=25.
  - [ ] Write formatters for `FileHandler` like `CommaSeparatedFormat`, `TrainerJournalFormat` etc.. maybe Journal is default for score cb.
    - [ ] where journal format is like "%(datetime): %(logger) %(level) - Epoch [1/50] Batch [220/1000(22%)] Accuracy=0.24 Loss=.024"
  - [ ] Write filters like `Score_TrainingStep` `Score_ValidationRun` etc. also their combination.
