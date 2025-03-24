"""
Project: Learn how to use PyTorch Lighnting on a simple toy problem
Author: Dr. Arrykrishna Mootoovaloo
Reference: PyTorch Lighnting website
Date: December 2022
"""
from ml_collections import ConfigDict
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules import CIFAR10DataModule
import torchvision


def data_module(cfg: ConfigDict):
    """The data module for the CIFAR 10 data

    Args:
        config (ConfigDict): the main configuration file.
    """
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(
                cfg.trans.crop_size, cfg.trans.padding),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization()
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization()
        ]
    )

    module = CIFAR10DataModule(
        data_dir=cfg.paths.data,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms
    )

    return module
