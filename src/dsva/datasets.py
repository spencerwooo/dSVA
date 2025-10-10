from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet


def create_imagenet_dataloader(
    root,
    transform,
    split="train",
    batch_size=32,
    shuffle=True,
    num_workers=4,
):
    dataset = ImageNet(root=root, split=split, transform=transform)
    print(f"Found {len(dataset)} images in {split} set under `{root}`")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def create_nips_dataloader(root, transform, batch_size):
    raise NotImplementedError("T.B.D.")
