"""PromptDA with a pretrained pyramid backbone and trainable DPT head.

This variant is intended for fair head-only fine-tuning against the DINOv2
baseline: the timm backbone is ImageNet-pretrained and frozen, while the DPT
head is optimized on the depth task.
"""

import os
from typing import Optional

import torch
import torch.nn as nn

from promptda.model.config import model_configs
from promptda.model.dpt import DPTHead
from promptda.utils.logger import Log


class PromptDA(nn.Module):
    use_bn = False
    use_clstoken = False
    output_act = "sigmoid"

    BACKBONES = ("swint", "swins", "convnextt", "convnexts")

    def __init__(
        self,
        encoder: str = "swint",
        ckpt_path: Optional[str] = None,
        dpt_variant: str = "pyramid_prompt_fpn",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if encoder not in self.BACKBONES:
            raise ValueError(
                f"Pyramid PromptDA supports {self.BACKBONES}, got encoder={encoder!r}. "
                "Use promptda_baseline.PromptDA for DINOv2 ViT encoders."
            )

        model_config = model_configs[encoder]
        self.encoder = encoder
        self.dpt_variant = dpt_variant

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "Pyramid PromptDA requires timm. Install it in your training "
                "environment before using a pyramid encoder."
            ) from exc

        self.pretrained = timm.create_model(
            model_config["timm_name"],
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            **model_config.get("backbone_kwargs", {}),
        )

        feature_info = self.pretrained.feature_info
        in_channels = [info["num_chs"] for info in feature_info]

        self.depth_head = DPTHead(
            nclass=1,
            in_channels=in_channels,
            features=model_config["features"],
            out_channels=model_config["out_channels"],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act=self.output_act,
            dpt_variant=self.dpt_variant,
        )

        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        self._freeze_backbone_only_train_head()

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        Log.info(f"Trainable: {trainable:,} / {total:,} params (pyramid DPT head only)")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        encoder: str = "swint",
        dpt_variant: str = "pyramid_prompt_fpn",
        **hf_kwargs,
    ):
        if hf_kwargs:
            Log.warn(f"Unused kwargs in pyramid from_pretrained: {list(hf_kwargs.keys())}")
        return cls(
            encoder=encoder,
            ckpt_path=pretrained_model_name_or_path,
            dpt_variant=dpt_variant,
            pretrained_backbone=True,
        )

    @property
    def model_config(self):
        return model_configs[self.encoder]

    def _freeze_backbone_only_train_head(self):
        for p in self.pretrained.parameters():
            p.requires_grad = False
        for p in self.depth_head.parameters():
            p.requires_grad = True
        self.pretrained.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.pretrained.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.depth_head.parameters() if p.requires_grad]

    def load_checkpoint(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            Log.warn(f"Checkpoint {ckpt_path} not found")
            return

        Log.info(f"Loading pyramid checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if all(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            Log.warn(f"Missing keys: {missing}")
        if unexpected:
            Log.warn(f"Unexpected keys: {unexpected}")

    def forward(self, x: torch.Tensor, prompt_depth: Optional[torch.Tensor] = None):
        assert prompt_depth is not None, "prompt_depth is required"

        prompt_depth, min_val, max_val = self.normalize(prompt_depth)
        h, w = x.shape[-2:]

        with torch.no_grad():
            self.pretrained.eval()
            features = self.pretrained((x - self._mean) / self._std)

        depth = self.depth_head(
            features,
            patch_h=h // 14,
            patch_w=w // 14,
            prompt_depth=prompt_depth,
            output_size=(h, w),
        )
        depth = self.denormalize(depth, min_val, max_val)
        return depth

    @torch.no_grad()
    def predict(self, image: torch.Tensor, prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    def normalize(self, prompt_depth: torch.Tensor):
        B = prompt_depth.shape[0]
        min_val = torch.quantile(
            prompt_depth.reshape(B, -1), 0.0, dim=1, keepdim=True
        )[:, :, None, None]
        max_val = torch.quantile(
            prompt_depth.reshape(B, -1), 1.0, dim=1, keepdim=True
        )[:, :, None, None]
        denom = (max_val - min_val).clamp_min(1e-6)
        prompt_depth = (prompt_depth - min_val) / denom
        return prompt_depth, min_val, max_val

    def denormalize(
        self,
        depth: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ):
        return depth * (max_val - min_val) + min_val
