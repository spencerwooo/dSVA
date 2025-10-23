import torchvision.transforms as T


def create_dsva_transforms() -> tuple[T.Compose, T.Normalize]:
    """
    Create data transforms and normalizations for the dSVA generator.

    Returns:
        A tuple containing the transform and normalize objects.
    """
    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform, normalize
