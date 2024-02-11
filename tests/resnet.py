import torch
import logging
import numpy as np
from torchutils.logging import CSVHandler, scoreFilterStep, formatter
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.datasets.transforms import ToDevice
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms


class VOCTransform(object):
    def __init__(self, num_labels, one_hot=True):
        assert isinstance(num_labels, int)
        assert isinstance(one_hot, bool)
        self.num_labels = num_labels
        self.one_hot = one_hot

    def __call__(self, img):
        img = np.array(img)
        img = [img == i for i in range(self.num_labels)]
        img = np.stack(img).astype(np.float32)
        if not self.one_hot:
            img = img.argmax(0)
        return torch.from_numpy(img)


target_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    VOCTransform(21, one_hot=False),
    ToDevice('cuda'),
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ToDevice('cuda'),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = VOCSegmentation(root='datasets/voc-segmentation/2012',
                          image_set='train', download=False,
                          target_transform=target_transform,
                          transform=transform, year='2012')


# Download ImageNet labels
model = TrainerModel(
    model='FCN_ResNet101',
    criterion='cross_entropy',
    optimizer='SGD',
    scheduler='ReduceLROnPlateau',
    lr=1e-6,
    momentum=0.80,
    weights='COCO_WITH_VOC_LABELS_V1'
)

model.model = model.model.to('cuda')


def extract_dict(fn):
    def wrapper(x):
        return fn(x)['out']
    return wrapper


model.model.forward = extract_dict(model.model.forward)


def args_to_detach(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.detach(), args), **kwds)
    return wrapper


def args_to_numpy(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.detach().cpu().numpy(), args), **kwds)
    return wrapper


def args_to_flatten(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.flatten(), args), **kwds)
    return wrapper


@args_to_flatten
@args_to_numpy
@args_to_detach
def intersection_over_union(output, target):
    return -(output - target)


trainer = Trainer(model=model, train_dataset=dataset, valid_dataset=dataset)
handler_csv = CSVHandler("deneme.csv", "linear_model_loss")
handler_str = logging.StreamHandler()
handler_str.addFilter(scoreFilterStep(1, 1))
handler_str.setFormatter(formatter)
etime = trainer.train(num_epochs=10,
                      batch_size=16,
                      handlers=[handler_str],
                      metrics=model.get_score_names(),
                      num_epochs_per_validation=0)
