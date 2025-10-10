import os

import torch

from dsva import Generator, ViT
from dsva.datasets import create_imagenet_dataloader
from dsva.utils import logger, progress


def train(
    dataset_root: str = "datasets/imagenet2012",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    epochs: int = 1,
    lr: float = 2e-4,
    eps: int = 10,
    model: str = "dino_vitb16",
    stride: int = 16,
    layer: int = 10,
    facet: str = "key",
    attn_layer: int = -1,
    device: str = "cuda",
    save_dir: str = "checkpoints",
):
    """Training script for dSVA's single model variant."""

    # setup experiment
    log = logger()
    run = f"dsva_{model}_e{epochs}_bs{batch_size}_eps{eps}_l{layer}_{facet}"
    run += f"_attn{attn_layer}" if attn_layer >= 0 else ""
    save_dir = os.path.join(save_dir, run)
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Starting run: {run}")

    # initialize training components
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    vit = ViT.from_pretrained(model, stride=stride).to(device)

    dataloader = create_imagenet_dataloader(
        root=dataset_root,
        transform=vit.transform,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    eps = eps / 255.0  # scale perturb from [0, 255] to [0, 1]

    # train loop
    for epoch in range(epochs):
        description = f"Epoch {epoch + 1}/{epochs}"
        for i, (img, _) in enumerate(progress(dataloader, description=description)):
            generator.train()
            optimizer.zero_grad()

            # forward pass
            img = img.to(device)
            adv = generator(img)

            # project adversarial perturbation to Linf ball
            delta = torch.clamp(adv - img, min=-eps, max=eps)
            adv = torch.clamp(img + delta, min=0, max=1)

            # img_feats, attn = vit.get_feats(
            img_feats = vit.get_feats(
                vit.normalize(img),
                layer=layer,
                facet=facet,
                attn_layer=attn_layer,
            )
            # adv_feats, _ = vit.get_feats(
            adv_feats = vit.get_feats(
                vit.normalize(adv),
                layer=layer,
                facet=facet,
                attn_layer=-1,  # only use benign image's attn
            )

            loss = torch.cosine_similarity(
                x1=img_feats.reshape(img_feats.shape[0], -1),
                x2=adv_feats.reshape(adv_feats.shape[0], -1),
                dim=-1,
            ).mean()

            loss.backward()
            optimizer.step()

            # log running loss
            if (i % 2000 == 0) or (i == len(dataloader) - 1):
                loss_log = (
                    f"epoch {epoch + 1}/{epochs}, "
                    f"step {i + 1}/{len(dataloader)}: loss {loss.item():.6f}"
                )
                log.info(loss_log)

        # save model checkpoint
        checkpoint_path = os.path.join(save_dir, f"generator_epoch{epoch + 1}.pth")
        torch.save(generator.state_dict(), checkpoint_path)
        log.info(f"Saved model to {checkpoint_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(train)
