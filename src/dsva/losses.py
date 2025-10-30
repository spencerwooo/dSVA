import torch
import torch.nn as nn

from dsva.model import ViT


class SVALoss(nn.Module):
    """
    Self-supervised ViT adversarial loss.

    This is the single model loss variant in dSVA, i.e., non-dual variant.

    Args:
        model: Pretrained ViT model loaded via `dsva.model.ViT`.
        layer: Layer index to extract features from.
        facet: Feature facet to extract. One of `key`, `query`, `value`, or `token`.
        attn_layer: Attention layer index to extract attention saliency maps from.
            Set to -1 to disable attention regularization.
    """

    def __init__(self, model: ViT, layer: int, facet: str, attn_layer: int):
        super(SVALoss, self).__init__()
        self.model = model

        self.layer = layer
        self.facet = facet
        self.attn_layer = attn_layer  # -1 for no attention

    def forward(self, img: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between benign and adversarial images.

        Args:
            img: Normalized benign images of shape (B, C, H, W).
            adv: Normalized adversarial images of shape (B, C, H, W).

        Returns:
            Loss value as a scalar tensor.
        """

        # acquire image and adv features, and benign image attention
        img_feats, attn = self.model.get_feats_and_attn(
            img, layer=self.layer, facet=self.facet, attn_layer=self.attn_layer
        )
        adv_feats, _ = self.model.get_feats_and_attn(
            adv, layer=self.layer, facet=self.facet, attn_layer=-1
        )

        # apply attention regularization if required
        if attn is not None:
            attn = attn.detach()  # (B, num_heads, N, N)
            attn = attn[:, :, 0, 1:]  # (B, num_heads, N-1)
            attn = attn.mean(dim=1, keepdim=True)  # (B, 1, N-1), average over heads
            attn = attn.unsqueeze(-1) * 100  # (B, 1, N-1, 1), scale up

            img_feats = img_feats * attn
            adv_feats = adv_feats * attn

        # compute loss
        loss = torch.cosine_similarity(
            x1=img_feats.reshape(img_feats.shape[0], -1),
            x2=adv_feats.reshape(adv_feats.shape[0], -1),
            dim=-1,
        ).mean()
        return loss
