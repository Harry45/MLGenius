"""
Project: Learn how to use PyTorch Lighnting on a simple toy problem
Author: Dr. Arrykrishna Mootoovaloo
Reference: PyTorch Lighnting website
Date: December 2022
"""
import os
import torch
from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Creates the configurations for running the main script.

    Returns:
        ConfigDict: all the configurations
    """

    cfg = ConfigDict()

    cfg.nclass = 10
    cfg.device = 1 if torch.cuda.is_available() else None

    # transformations
    cfg.trans = trans = ConfigDict()
    trans.crop_size = 32
    trans.padding = 4

    # training
    cfg.training = training = ConfigDict()
    training.batch_size = 4
    training.num_workers = int(os.cpu_count() / 2)
    training.nepochs = 1

    # paths
    os.makedirs('data/', exist_ok=True)
    os.makedirs('logs/', exist_ok=True)
    cfg.paths = paths = ConfigDict()
    paths.data = 'data/'
    paths.logs = 'logs/'

    return cfg
