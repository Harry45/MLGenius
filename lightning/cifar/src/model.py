"""
Project: Learn how to use PyTorch Lighnting on a simple toy problem
Author: Dr. Arrykrishna Mootoovaloo
Reference: PyTorch Lighnting website
Date: December 2022
"""
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule
from ml_collections import ConfigDict


def create_model(cfg: ConfigDict) -> torchvision.models.resnet.ResNet:
    """Create the deep learning model to use.

    Args:
        cfg (ConfigDict): the main configuration file.

    Returns:
        torchvision.models.resnet.ResNet: a ResNet model with a few additional layers
    """
    model = torchvision.models.resnet18(
        pretrained=False, num_classes=cfg.nclass)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=(
        3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    """Lightning module for the MNIST example

    Args:
        cfg (ConfigDict): the set of configurations for the model.
    """

    def __init__(self, cfg: ConfigDict, lr=0.05):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = create_model(self.cfg)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """The image is passed through the model.

        Args:
            image (torch.Tensor): the input image.

        Returns:
            torch.Tensor: the output from the model
        """
        out = self.model(image)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """The training step in the lightning module.

        Args:
            batch (list): a list containing the images and labels.
            batch_idx (int): the index of the batch - not used but maybe important for lightning?

        Returns:
            torch.Tensor: the value of the loss function.
        """
        image, label = batch
        logits = self(image)
        loss = F.nll_loss(logits, label)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch: list, stage: str = None):
        """Evaluate the model on unseen data.

        Args:
            batch (list): a list containing the images and labels.
            stage (str, optional): Evaluation stage: train, test, val. Defaults to None.
        """
        image, label = batch
        logits = self(image)
        loss = F.nll_loss(logits, label)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, label, task="multiclass",
                       num_classes=self.cfg.nclass)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch: list, batch_idx: int):
        """The validation step where we evaluate the model.

        Args:
            batch (list): a list containing the images and labels.
            batch_idx (int): the index of the batch.
        """
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        """The validation step where we evaluate the model in the test stage.

        Args:
            batch (list): a list containing the images and labels.
            batch_idx (int): the index of the batch.
        """
        self.evaluate(batch, "test")

    def configure_optimizers(self) -> dict:
        """Configure the optimizer

        Returns:
            dict: dictionary containing the optimizer and scheduler.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.cfg.training.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
