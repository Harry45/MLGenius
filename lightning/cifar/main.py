"""
Project: Learn how to use PyTorch Lighnting on a simple toy problem
Author: Dr. Arrykrishna Mootoovaloo
Reference: PyTorch Lighnting website
Date: December 2022
"""

import torch
from absl import flags, app
from ml_collections.config_flags import config_flags
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

# our scripts
from src.processing import data_module
from src.model import LitResnet

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)


def main(argv):
    module = data_module(FLAGS.config)
    model = LitResnet(FLAGS.config)

    cbacks = [LearningRateMonitor(logging_interval="step"),
              TQDMProgressBar(refresh_rate=10)]
    logger = CSVLogger(save_dir=FLAGS.config.paths.logs)

    trainer = Trainer(
        max_epochs=FLAGS.config.training.nepochs,
        accelerator="auto",
        devices=FLAGS.config.device,
        logger=logger,
        callbacks=cbacks,
    )

    trainer.fit(model, module)
    trainer.test(model, datamodule=module)


if __name__ == "__main__":
    app.run(main)
