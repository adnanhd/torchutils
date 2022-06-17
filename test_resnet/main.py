import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torchutils.trainer import Trainer
from torchutils.utils.pydantic import TrainerModel
from torchutils.callbacks import ProgressBar

# Download ImageNet labels

# Read the categories
with open("test_resnet/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

batch_size = 64

TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(
    root='./test_resnet/data', train=True, download=True, transform=TF)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./test_resnet/data', train=False, download=True, transform=TF)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

CIFAR10_classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# For reproducibility, let us recreate the FC layer here with a fixed seed:
torch.manual_seed(501)
random.seed(501)
np.random.seed(501)


def get_learnable_parameters(model):
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update


resnet18 = models.resnet18(pretrained=True)

for param in resnet18.parameters():
    param.requires_grad_(False)

resnet18.fc = nn.Linear(512, 10)
params_to_update = get_learnable_parameters(resnet18)

model = TrainerModel(
    model=resnet18,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.SGD(params_to_update, lr=0.005, momentum=0.80)
)
trainer = Trainer(model=model, device='cuda', ytype=torch.long)
pbar = ProgressBar()
trainer.compile(callbacks=[pbar])
# trainer.compile(metrics=['loss'])
# trainer.compile(loggers=[pbar._step_bar])

trainer.train(num_epochs=10, batch_size=batch_size, train_dataset=trainset)
