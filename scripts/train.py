from dataclasses import dataclass

import torch

from dsva.dataloader import create_imagenet_dataloader
from dsva.generator import Generator
from dsva.losses import SVALoss
from dsva.model import ViT
from dsva.trainer import Trainer
from dsva.transforms import create_dsva_transforms
from dsva.utils import get_timestamp, setup_run


@dataclass
class ModelConfig:
    """
    Model configuration.

    Args:
        name: Surrogate ViT model name to use for feature extraction. One of
            `dino_vitb16`, `dino_vits16`, `dino_vits8`, `dino_vitb8`, `mae_vitb16`,
            `vits16`, `vits8`, `vitb16`, or `vitb8`.
        stride: Patch stride for the ViT model.
        layer: Layer used within the ViT model blocks.
        facet: Q/K/V facet to use. One of `key`, `query`, `value`, or `token`.
        attn_layer: Attention layer to use for extracting the attention saliency
            map. If set as lower than 0, no attention regularization is applied.
    """

    name: str = "dino_vitb16"
    stride: int = 16
    layer: int = 10
    facet: str = "key"
    attn_layer: int = -1


@dataclass
class DataConfig:
    """
    Data configuration.

    Args:
        root: Path to the ImageNet2012 dataset root directory.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory for data loading.
    """

    root: str = "datasets/imagenet2012"
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """
    Training configuration.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate for the Adam optimizer.
        log_every_n_steps: Log running loss every number of steps.
    """

    epochs: int = 1
    lr: float = 2e-4
    log_every_n_steps: int = 2000


def train(
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = "outputs/runs",
    eps: int = 10,
    model: ModelConfig = ModelConfig(),
    data: DataConfig = DataConfig(),
    train: TrainConfig = TrainConfig(),
) -> None:
    """
    Training script for dSVA's single model variant.

    Args:
        seed: Random seed for reproducibility.
        device: Device to use for training. Either `cuda` or `cpu`.
        save_dir: Directory to save model checkpoints and other training intermediates.
        eps: Max perturbation in Linf norm (in [0, 255] scale).
        model: Model configuration.
        data: Data configuration.
        train: Training configuration.
    """

    # setup experiment
    run_id = f"dsva_{model.name}_ep{train.epochs}_bs{data.batch_size}_eps{eps}_l{model.layer}_{model.facet}"
    run_id += f"_attn{model.attn_layer}" if model.attn_layer >= 0 else ""
    run_dir = setup_run(seed, save_dir, f"{run_id}_{get_timestamp()}", args=locals())

    # initialize training components
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    eps = eps / 255.0  # scale perturbation from [0, 255] to [0, 1]

    generator = Generator().to(device)
    vit = ViT.from_pretrained(model.name, stride=model.stride).to(device)
    transform, normalize = create_dsva_transforms()

    # construct optimizer and dSVA loss function
    optimizer = torch.optim.Adam(generator.parameters(), lr=train.lr)
    criterion = SVALoss(
        model=vit,
        layer=model.layer,
        facet=model.facet,
        attn_layer=model.attn_layer,
    )
    dataloader = create_imagenet_dataloader(
        root=data.root,
        transform=transform,
        batch_size=data.batch_size,
        shuffle=data.shuffle,
        num_workers=data.num_workers,
        pin_memory=data.pin_memory,
    )

    # train loop
    trainer = Trainer(generator, normalize, eps, optimizer, criterion, device)
    trainer.train(
        dataloader,
        run_dir,  # saves to "{run_dir}/generator_epoch_{epoch}.pth"
        num_epochs=train.epochs,
        log_every_n_steps=train.log_every_n_steps,
    )


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(train)
