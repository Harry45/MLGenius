# Implementation of the Vision Transformer (ViT) model
# https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange


MODEL_PATH = "models/vit_model.pth"  # Path to save/load model
TRAIN_MODEL = True  # Set to False to load a pre-trained model instead

def patchify(images: Tensor, n_patches: int) -> Tensor:
    """Splits a batch of square images into non-overlapping patches.

    Args:
        images (Tensor): A tensor of shape (N, C, H, W), where
                        - N is the batch size,
                        - C is the number of channels, and
                        - H, W are the spatial dimensions.
        n_patches (int): The number of patches per dimension. The total patches per image
            will be n_patches^2.

    Returns:
        Tensor: A tensor of shape (N, n_patches^2, patch_size * patch_size * C) containing
        flattened image patches.

    Raises:
        ValueError: If the images are not square.
    """
    n, c, h, w = images.shape

    if h != w:
        raise ValueError("Patchify method is implemented for square images only")

    patch_size = h // n_patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n, n_patches**2, -1)
    return patches


class MultiheadSelfAttention(nn.Module):
    """Implements a simple Multi-Head Self-Attention (MSA) mechanism.

    Args:
        embed_dim (int): The dimensionality of input token embeddings.
        num_heads (int): The number of attention heads (default: 2).

    Raises:
        ValueError: If `embed_dim` is not divisible by `num_heads`.
    """
    def __init__(self, embed_dim: int, num_heads: int = 2) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_mappings = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences: Tensor) -> Tensor:
        """Computes multi-head self-attention over input sequences.

        Args:
            sequences (Tensor): Input tensor of shape (N, seq_length, embed_dim).

        Returns:
            Tensor: Output tensor of shape (N, seq_length, embed_dim) after self-attention.
        """
        batch_results = []
        for sequence in sequences:  # Process each sequence in the batch
            head_results = []
            for head in range(self.num_heads):
                q = self.q_mappings[head](sequence[:, head * self.head_dim : (head + 1) * self.head_dim])
                k = self.k_mappings[head](sequence[:, head * self.head_dim : (head + 1) * self.head_dim])
                v = self.v_mappings[head](sequence[:, head * self.head_dim : (head + 1) * self.head_dim])

                attention_scores = self.softmax(q @ k.T / (self.head_dim ** 0.5))
                head_results.append(attention_scores @ v)

            batch_results.append(torch.hstack(head_results))

        return torch.stack(batch_results)

class VisionTransformerBlock(nn.Module):
    """Implements a Transformer block used in Vision Transformers (ViT).

    Args:
        embed_dim (int): The dimensionality of input embeddings.
        num_heads (int): The number of attention heads.
        mlp_ratio (int, optional): Expansion factor for the MLP layer (default: 4).
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = MultiheadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Processes input tensor through the Vision Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            Tensor: Output tensor of the same shape as input after processing.
        """
        x = x + self.mhsa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def get_positional_embeddings(sequence_length: int, embed_dim: int) -> Tensor:
    """Generates sinusoidal positional embeddings."""
    position = torch.arange(sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
    positional_embedding = torch.zeros(sequence_length, embed_dim)
    positional_embedding[:, 0::2] = torch.sin(position * div_term)
    positional_embedding[:, 1::2] = torch.cos(position * div_term)
    return positional_embedding

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) implementation for image classification.

    Args:
        input_shape (tuple): Shape of input images (C, H, W).
        num_patches (int, optional): Number of patches per dimension (default: 7).
        num_blocks (int, optional): Number of Transformer blocks (default: 2).
        embed_dim (int, optional): Embedding dimension (default: 8).
        num_heads (int, optional): Number of attention heads (default: 2).
        num_classes (int, optional): Number of output classes (default: 10).
    """
    def __init__(
        self,
        input_shape: tuple,
        num_patches: int = 7,
        num_blocks: int = 2,
        embed_dim: int = 8,
        num_heads: int = 2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.num_patches = num_patches
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if input_shape[1] % num_patches != 0 or input_shape[2] % num_patches != 0:
            raise ValueError("Input shape must be divisible by number of patches")

        self.patch_size = (input_shape[1] // num_patches, input_shape[2] // num_patches)
        self.input_dim = input_shape[0] * self.patch_size[0] * self.patch_size[1]
        self.linear_mapper = nn.Linear(self.input_dim, embed_dim)

        self.class_token = nn.Parameter(torch.rand(1, embed_dim))
        self.register_buffer("positional_embeddings",
                             get_positional_embeddings(num_patches**2 + 1, embed_dim),
                             persistent=False)
        self.blocks = nn.ModuleList([VisionTransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)])

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, images: Tensor) -> Tensor:
        """Processes input images and outputs class probabilities.

        Args:
            images (Tensor): Input tensor of shape (batch_size, C, H, W).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes) with class probabilities.
        """
        batch_size, _, _, _ = images.shape
        patches = patchify(images, self.num_patches).to(self.positional_embeddings.device)
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(batch_size, 1, -1), tokens), dim=1)
        out = tokens + self.positional_embeddings.repeat(batch_size, 1, 1)


        for block in self.blocks:
            out = block(out)

        return self.mlp_head(out[:, 0])

def load_data(batch_size: int = 128):
    """Loads MNIST dataset and returns train and test data loaders."""
    transform = ToTensor()
    train_set = MNIST(root="./datasets", train=True, download=True, transform=transform)
    test_set = MNIST(root="./datasets", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader

def initialize_model(device: torch.device):
    """Initializes the Vision Transformer model and moves it to the given device."""
    model = VisionTransformer((1, 28, 28),
                              num_patches=7,
                              num_blocks=2,
                              embed_dim=8,
                              num_heads=2,
                              num_classes=10).to(device)
    return model

def train_model(model, train_loader, device, epochs: int = 5, lr: float = 0.005):
    """Trains the model using the given data loader."""
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for epoch in trange(epochs, desc="Training"):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, device):
    """Evaluates the trained model on the test set."""
    model.eval()
    criterion = CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            total_loss += criterion(y_hat, y).item()
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total * 100

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")


def main():
    """Main function to train and evaluate the Vision Transformer model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} {'('+torch.cuda.get_device_name(device)+')' if torch.cuda.is_available() else ''}")

    # Load dataset
    train_loader, test_loader = load_data()

    # Initialize model
    model = initialize_model(device)

    if TRAIN_MODEL or not os.path.exists(MODEL_PATH):
        print("\n Training model...")
        train_model(model, train_loader, device)

        # Save the trained model
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\n Model saved to {MODEL_PATH}")
    else:
        # Load model from saved checkpoint
        print(f"\n Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()