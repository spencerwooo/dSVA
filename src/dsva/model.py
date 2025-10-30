import math
import types
from typing import Callable

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from dsva.utils import get_logger

log = get_logger(__name__)

timm_name_mappings = {
    "mae_vitb16": "timm/vit_base_patch16_224.mae",
    "vits16": "timm/vit_small_patch16_224",
    "vits8": "timm/vit_small_patch8_224",
    "vitb16": "timm/vit_base_patch16_224",
    "vitb8": "timm/vit_base_patch8_224",
}
dino_arch_mappings = {
    "mae_vitb16": "dino_vitb16",
    "vits16": "dino_vits16",
    "vits8": "dino_vits8",
    "vitb16": "dino_vitb16",
    "vitb8": "dino_vitb8",
}


class ViT:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        transform: Callable[[Image.Image, torch.Tensor], torch.Tensor],
        normalize: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.name = name
        self.model = model.eval()
        self.transform = transform
        self.normalize = normalize

        # essential attributes
        self.p = self.model.patch_embed.patch_size
        self.stride = self.model.patch_embed.proj.stride
        log.info(f"Loaded `{name}` with patch size {self.p} and stride {self.stride}")

        # intermediate features keyed by (layer_index, facet)
        self._feats: dict[tuple[int, str], torch.Tensor] = {}

        # internal module hooks
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.load_size = None
        self.num_patches = None

    @classmethod
    def from_pretrained(cls, name: str, stride: int = 16) -> "ViT":
        """
        Initialize a ViT model (DINO, MAE, or supervised ViT models) from pretrained
        weights sourced from either torch.hub or timm.

        N.B.:
            We may not use the transform and normalizations resolved from their
            metadata, as a separate preprocessing is in `src/dsva/transforms.py` to
            account for training the generator with dual models.

        Args:
            name: Name of the model. One of `dino_vitb16`, `dino_vits16`, `dino_vits8`,
                `dino_vitb8`, `mae_vitb16`, `vits16`, `vits8`, `vitb16`, or `vitb8`.

        Returns:
            The pretrained ViT model with preprocessing transform and normalizations.
        """

        def _load_dino_model(name: str) -> nn.Module:
            return torch.hub.load("facebookresearch/dino:main", name)

        # load DINO model and pretrained weights directly from torch.hub
        if "dino" in name:
            model = _load_dino_model(name)

            # DINO's preprocessing parameters come from
            # https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/eval_knn.py#L32-L37
            transform = T.Compose(
                [
                    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]
            )
            normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # load pretrained MAE or supervised ViT weights from timm
        else:
            timm_model = timm.create_model(timm_name_mappings[name], pretrained=True)
            timm_weights = timm_model.state_dict()

            # remove head weights for supervised ViT models
            if "mae" not in name:
                del timm_weights["head.weight"]
                del timm_weights["head.bias"]

            # use the DINO architecture for MAE and supervised ViT models
            model = _load_dino_model(dino_arch_mappings[name])
            # load pretrained weights separately
            model.load_state_dict(timm_weights)

            # resolve preprocessing methods
            cfg = timm.data.resolve_data_config(timm_model.pretrained_cfg)
            normalize = T.Normalize(mean=cfg["mean"], std=cfg["std"])
            transform = timm.data.create_transform(**cfg, is_training=False)
            transform.transforms = [
                tr for tr in transform.transforms if not isinstance(tr, T.Normalize)
            ]

        # patch model's resolution if needed
        model = patch_vit_resolution(model, stride=stride)
        return cls(name=name, model=model, transform=transform, normalize=normalize)

    def to(self, device: torch.device) -> "ViT":
        self.model.to(device)
        return self

    def get_feats_and_attn(
        self,
        x: torch.Tensor,
        layer: int,
        facet: str,
        attn_layer: int,
        include_cls: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Extract features and raw attention from specified layers and facets.

        Args:
            x: Input images of shape (B, C, H, W).
            layer: Layer index to extract features from.
            facet: Feature facet to extract. One of `key`, `query`, `value`, or `token`.
            attn_layer: Attention layer index to extract raw attention from. Set to -1
                to disable attention extraction.
            include_cls: Whether to include the CLS token in the extracted features.

        Returns:
            A tuple of:
            - Extracted features of shape (B, 1, N, D), where N is the number of patches
              (including CLS token if `include_cls` is True) and D is the feature dim.
            - Extracted raw attention of shape (B, num_heads, N, N) if `attn_layer` >=
              0, else None.
        """

        supported_facets = ["key", "query", "value", "token"]
        assert facet in supported_facets, (
            f"facet {facet} not supported, choose from {', '.join(supported_facets)}"
        )

        # clear previous features
        self._feats = {}

        # register hooks for requested facet and attention (if any)
        B, C, H, W = x.shape
        self._register_hooks(layers=[layer], facet=facet)
        if attn_layer >= 0:
            self._register_hooks(layers=[attn_layer], facet="attn")

        # forward pass
        self.model(x)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )

        # acquire features and optional attention maps
        feats = self._feats[(layer, facet)]  # Bxhxtxd
        attn = self._feats[(attn_layer, "attn")] if attn_layer >= 0 else None

        # post-process facet-level features
        if facet == "token":
            feats.unsqueeze_(1)  # (B, 1, t, head_dim)
        if not include_cls:
            feats = feats[:, :, 1:, :]  # (B, num_heads, t-1, head_dim)
        feats = feats.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)
        feats = feats.unsqueeze(dim=1)

        return feats, attn

    def _create_hook(self, layer: int, facet: str) -> Callable:
        """Generate a hook method for a specific block and facet."""
        # for entire layer output facets, i.e., "attn" and "token"
        if facet in ["attn", "token"]:

            def _hook(m, i, o):
                self._feats[(layer, facet)] = o

            return _hook

        # for inner qkv facets, i.e., "key", "query", and "value"
        facet_idx_map = {"query": 0, "key": 1, "value": 2}
        facet_idx = facet_idx_map[facet]

        def _inner_hook(m, i, o):
            i = i[0]
            B, N, C = i.shape
            qkv = (
                m.qkv(i)
                .reshape(B, N, 3, m.num_heads, C // m.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats[(layer, facet)] = qkv[facet_idx]  # (B, heads, N, head_dim)

        return _inner_hook

    def _register_hooks(self, layers: list[int], facet: str) -> None:
        """
        Register hooks to extract features.

        N.B.: Set `layers` as a list to support multiple layer extraction in the future.

        Args:
            layers: Layers from which to extract features.
            facet: Facet to extract. One of `key`, `query`, `value`, `token`, or `attn`.
        """

        for layer, block in enumerate(self.model.blocks):
            if layer not in layers:
                continue
            module = {
                "key": block.attn,
                "query": block.attn,
                "value": block.attn,
                "token": block,
                "attn": block.attn.attn_drop,
            }
            h = module[facet].register_forward_hook(self._create_hook(layer, facet))
            self.hooks.append(h)

    def _unregister_hooks(self) -> None:
        """Unregisters the hooks, should be called after feature extraction."""
        for h in self.hooks:
            h.remove()
        self.hooks = []


def fix_pos_enc(patch_size: int, stride_hw: tuple[int, int]) -> Callable:
    """
    Args:
        patch_size: The patch size of the model.
        stride_hw: A tuple containing the new height and width stride respectively.

    Returns:
        The interpolation method.
    """

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert w0 * h0 == npatch, (
            f"got wrong grid size for {h}x{w} with patch_size {patch_size} and "
            f"stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"
        )
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding


def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
    """
    Args:
        model (nn.Module): The model to change resolution for.
        stride (int): The new stride parameter.

    Returns:
        nn.Module: The adjusted model.
    """

    patch_size = model.patch_embed.patch_size
    if stride == patch_size:  # nothing to do
        return model

    # stride = nn_utils._pair(stride)
    stride = nn.modules.utils._pair(stride)
    assert all([(patch_size // s_) * s_ == patch_size for s_ in stride]), (
        f"stride {stride} should divide patch_size {patch_size}"
    )

    # fix the stride
    model.patch_embed.proj.stride = stride
    # fix the positional encoding code
    model.interpolate_pos_encoding = types.MethodType(
        fix_pos_enc(patch_size, stride), model
    )
    return model


if __name__ == "__main__":
    vit = ViT.from_pretrained("dino_vitb16")
    print(
        f"{vit.name=}",
        f"{vit.p=}",
        f"{vit.stride=}",
        f"{vit.transform=}",
        f"{vit.normalize=}",
        sep="\n",
    )

    x = torch.randn(1, 3, 224, 224)
    out = vit.model(x)
    print(f"{out.shape=}")  # should be [1, 768]

    feats, attn = vit.get_feats_and_attn(x, layer=10, facet="key", attn_layer=-1)
    print(f"{feats.shape=}")  # should be [1, 1, 196, 64*12]
    print(f"{attn is None=}")  # should be True when attn_layer < 0
