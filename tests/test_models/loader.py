import torch
import os.path as osp
import torchutils.datasets as td
import torchvision.datasets as tvd
import torchvision.transforms as tf


def get_cifar_transforms(mean, std, image_size=32, random_resized_crop=False, random_crop=False, random_flip=False, **kwds):
    train_tf = list()
    valid_tf = list()

    if random_resized_crop:
        train_tf.append(
            tf.RandomResizedCrop( 
                size=image_size, scale=kwds.get('scale', (0.8, 1.0)), ratio=kwds.get('ratio', (0.9, 1.1))
            )
        )
    else:
        if image_size != 32 or image_size != (32, 32):
            train_tf.append(tf.Resize(image_size))

        if random_crop:
            train_tf.append(tf.RandomCrop(image_size, padding=kwds.get('padding', image_size // 8)))

    if random_flip:
        train_tf.append(tf.RandomHorizontalFlip())

    if image_size != 32 or image_size != (32, 32):
        valid_tf.append(tf.Resize(image_size))

    cifar10_tf = [tf.ToTensor(), tf.Normalize(mean, std)]

    return tf.Compose(train_tf + cifar10_tf), tf.Compose(valid_tf + cifar10_tf)


@td.register_builder('cifar-10')
def cifar10_builder(prefix='data', image_size=32, random_resized_crop=True, random_crop=False, random_flip=False, normalize=True, **kwds):
    assert image_size in [32, 224, 384]

    DATASET_PATH = osp.join(prefix, 'cifar')
    CIFAR10_TRAIN_STD = [0.24703233, 0.24348505, 0.26158768]
    CIFAR10_TRAIN_MEAN = [0.49139968, 0.48215827, 0.44653124]

    train_tf, valid_tf = get_cifar_transforms(mean=CIFAR10_TRAIN_MEAN if normalize else 0,
                                              std=CIFAR10_TRAIN_STD if normalize else 1,
                                              image_size=image_size, random_crop=random_crop,
                                              random_resized_crop=random_resized_crop,
                                              random_flip=random_flip, **kwds)

    return (tvd.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_tf), 
            tvd.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=valid_tf))


@td.register_builder('cifar-100')
def cifar100_builder(prefix='data', image_size=32, random_resized_crop=True, random_crop=False, random_flip=False, normalize=True, **kwds):
    assert image_size in [32, 224, 384]

    DATASET_PATH = osp.join(prefix, 'cifar-100')
    CIFAR100_TRAIN_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR100_TRAIN_STD = [0.2023, 0.1994, 0.2010]

    train_tf, valid_tf = get_cifar_transforms(mean=CIFAR100_TRAIN_MEAN if normalize else 0,
                                              std=CIFAR100_TRAIN_STD if normalize else 1,
                                              image_size=image_size, random_crop=random_crop,
                                              random_resized_crop=random_resized_crop,
                                              random_flip=random_flip, **kwds)

    return (tvd.CIFAR100(root=DATASET_PATH, train=True, download=True, transform=train_tf), 
            tvd.CIFAR100(root=DATASET_PATH, train=False, download=True, transform=valid_tf))


def compute_mean_std(dataset):
    dataset = td.WrapDataset(dataset)
    channels = dataset.apply(lambda dataset: torch.stack([image for image, _ in dataset], dim=1))
    return dict(mean=channels.mean((1,2,3)), std=channels.std((1, 2, 3)))
