import os
from dataclasses import dataclass

import torch

from dsva import Generator, ViT
from dsva.dataloader import create_imagenet_dataloader
from dsva.utils import get_timestamp, progress, setup_run


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
        log_every_n_steps: Frequency of logging the training loss.
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
    run_dir, logger = setup_run(seed, save_dir, f"{run_id}_{get_timestamp()}", locals())

    # initialize training components
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    optim = torch.optim.Adam(generator.parameters(), lr=train.lr, betas=(0.5, 0.999))
    vit = ViT.from_pretrained(model.name, stride=model.stride).to(device)

    dataloader = create_imagenet_dataloader(
        root=data.root,
        transform=vit.transform,
        batch_size=data.batch_size,
        shuffle=data.shuffle,
        num_workers=data.num_workers,
        pin_memory=data.pin_memory,
    )
    eps = eps / 255.0  # scale perturbation from [0, 255] to [0, 1]

    # train loop
    for epoch in range(train.epochs):
        description = f"Epoch {epoch + 1}/{train.epochs}"
        for i, (img, _) in enumerate(progress(dataloader, desc=description)):
            generator.train()
            optim.zero_grad()

            # forward pass
            img = img.to(device)
            adv = generator(img)

            # project adversarial perturbation to Linf ball
            delta = torch.clamp(adv - img, min=-eps, max=eps)
            adv = torch.clamp(img + delta, min=0, max=1)

            # img_feats, attn = vit.get_feats(
            img_feats = vit.get_feats(
                vit.normalize(img),
                layer=model.layer,
                facet=model.facet,
                attn_layer=model.attn_layer,
            )
            # adv_feats, _ = vit.get_feats(
            adv_feats = vit.get_feats(
                vit.normalize(adv),
                layer=model.layer,
                facet=model.facet,
                attn_layer=-1,  # only use benign image's attn
            )

            loss = torch.cosine_similarity(
                x1=img_feats.reshape(img_feats.shape[0], -1),
                x2=adv_feats.reshape(adv_feats.shape[0], -1),
                dim=-1,
            ).mean()

            loss.backward()
            optim.step()

            # log running loss
            if (i % train.log_every_n_steps == 0) or (i == len(dataloader) - 1):
                loss_log = (
                    f"epoch {epoch + 1}/{train.epochs}, "
                    f"step {i + 1}/{len(dataloader)}: loss {loss.item():.6f}"
                )
                logger.info(loss_log)

        # save model checkpoint each epoch
        checkpoint_path = os.path.join(run_dir, f"generator_epoch{epoch + 1}.pth")
        torch.save(generator.state_dict(), checkpoint_path)
        logger.info(f"Saved model to {checkpoint_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(train)
