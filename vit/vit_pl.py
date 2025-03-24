import os
import torch
import torch.nn as nn
import lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from vit_torch import VisionTransformer
from typing import Optional, Tuple

# Model path and training flag
MODEL_PATH: str = "models/vit_pl_model.pth"
TRAIN: bool = True


class ViTLightning(pl.LightningModule):
    """A LightningModule wrapper for the Vision Transformer (ViT) model.

    Attributes:
        model (VisionTransformer): The Vision Transformer model.
        criterion (CrossEntropyLoss): The loss function.
        lr (float): Learning rate for optimization.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        num_patches: int = 7,
        num_blocks: int = 2,
        embed_dim: int = 8,
        num_heads: int = 2,
        num_classes: int = 10,
        lr: float = 0.005,
    ) -> None:
        """Initializes the Vision Transformer model.

        Args:
            input_shape (Tuple[int, int, int]): Input image shape (C, H, W).
            num_patches (int): Number of patches for ViT.
            num_blocks (int): Number of transformer blocks.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            num_classes (int): Number of output classes.
            lr (float): Learning rate.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model: VisionTransformer = VisionTransformer(
            input_shape, num_patches, num_blocks, embed_dim, num_heads, num_classes
        )
        self.criterion: CrossEntropyLoss = CrossEntropyLoss()
        self.lr: float = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Performs a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Performs a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Performs a single test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """Configures the optimizer for training.

        Returns:
            Optimizer: Adam optimizer.
        """
        return Adam(self.parameters(), lr=self.lr)


class MNISTDataModule(pl.LightningDataModule):
    """A LightningDataModule for the MNIST dataset.

    Attributes:
        batch_size (int): Batch size for training.
        transform (ToTensor): Transformation applied to the dataset.
    """

    def __init__(self, batch_size: int = 128) -> None:
        """Initializes the MNIST DataModule.

        Args:
            batch_size (int): Batch size for training.
        """
        super().__init__()
        self.batch_size: int = batch_size
        self.transform: ToTensor = ToTensor()
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepares the dataset for training, validation, and testing.

        Args:
            stage (Optional[str]): Stage identifier ("fit", "test", etc.).
        """
        dataset = MNIST(root="./datasets", train=True, download=True, transform=self.transform)
        self.train_set, self.val_set = random_split(dataset, [55000, 5000])
        self.test_set = MNIST(root="./datasets", train=False, download=True, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for training.

        Returns:
            DataLoader: Training data loader.
        """
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=7, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for validation.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=7, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for testing.

        Returns:
            DataLoader: Test data loader.
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=7, pin_memory=True)


def main() -> None:
    """Main function to train or evaluate the Vision Transformer model."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ViTLightning()
    data_module = MNISTDataModule()

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator=device,
        log_every_n_steps=10,
    )

    if TRAIN:
        print("\nTraining model...")
        trainer.fit(model, data_module)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
    else:
        print(f"\nLoading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print("\nEvaluating model...")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
