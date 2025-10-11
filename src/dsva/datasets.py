from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from dsva.utils import logger

log = logger()


def create_imagenet_dataloader(
    root: str,
    transform: Callable[[Image.Image, torch.Tensor], torch.Tensor],
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a dataloader for the ImageNet2012 dataset.

    Args:
        root: Path to the ImageNet2012 dataset root directory.
        transform: Transform to apply to the images.
        split: Dataset split to use. Either `train` or `val`.
        batch_size: Batch size for the dataloader.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of workers for data loading.
    """
    dataset = ImageNet(root=root, split=split, transform=transform)
    log.info(f"Found {len(dataset)} images in {split} set under `{root}`")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def create_nips_dataloader(root, transform, batch_size):
    raise NotImplementedError("T.B.D.")
