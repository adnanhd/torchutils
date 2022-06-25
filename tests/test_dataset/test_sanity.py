from torchutils.trainer import Trainer
from torchutils.callbacks import ProgressBar
from torchutils.trainer import TrainerModel
import torch, numpy as np, collections
from torchutils.models import Convolution, FeedForward
from torchutils.data import Dataset
from torchutils.metrics import MetricHandler
from torchutils.data.transform import MeanStdTransform, MinMaxTransform

x = np.random.randn(1000, 100, 100)
y = (x * 10).astype(int).mean(2).std(1).reshape(1000, 1)

dataset = Dataset(features=x, labels=y)
model = Convolution(1, 4, 10, kernel_size=2)
model2 = Convolution(10, 20, 40, kernel_size=5)

1/0
model = torch.nn.Sequential(collections.OrderedDict())

#transform = MeanStdTransform(dataset.labels, axis=(0, ))
trainer = Trainer(model=model, device='cuda', 
        xtype=torch.float64, ytype=torch.float64)


transform.normalize()
transform.to_tensor(device=trainer.device)

valid_dl = dataset.split(20)

pbar = ProgressBar()
trainer.compile(callbacks=[pbar], loggers=[pbar._step_bar], metrics=['mse', 'mass', 'momx', 'momy'])
trainer.train(num_epochs=100, batch_size=16, learning_rate=1e-4, train_dataset=dataset)
transform.denormalize()

