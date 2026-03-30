"""
promptda/promptda.py

Two modes:
  Baseline (use_mlf=False): freeze all, zero-shot eval
  Experiment (use_mlf=True): freeze DINOv2, train MLF projector
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from promptda.model.config import model_configs
from promptda.model.dpt import DPTHead
from promptda.model.masked_fusion import MaskedLocalFusion
from promptda.utils.logger import Log


class PromptDA(nn.Module):

    patch_size   = 14
    use_bn       = False
    use_clstoken = False
    output_act   = 'sigmoid'

    HF_REPOS = {
        "vits": "depth-anything/prompt-depth-anything-vits",
        "vitb": "depth-anything/prompt-depth-anything-vitb",
        "vitl": "depth-anything/prompt-depth-anything-vitl",
    }

    def __init__(
        self,
        encoder: str = 'vitl',
        ckpt_path: Optional[str] = None,
        use_mlf: bool = True,
    ):
        super().__init__()
        self.encoder    = encoder
        self.use_mlf    = use_mlf
        model_config    = model_configs[encoder]

        # ── Backbone: DINOv2 ──────────────────────────────────────────────
        module_path      = Path(__file__)
        package_base_dir = str(Path(*module_path.parts[:-2]))
        self.pretrained  = torch.hub.load(
            f'{package_base_dir}/torchhub/facebookresearch_dinov2_main',
            'dinov2_{:}14'.format(encoder),
            source='local',
            pretrained=False,
        )
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        # ── Decoder: DPT head ─────────────────────────────────────────────
        self.depth_head = DPTHead(
            nclass=1,
            in_channels=dim,
            features=model_config['features'],
            out_channels=model_config['out_channels'],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act=self.output_act,
        )

        # ── MLF (always init to keep state_dict shape stable) ─────────────
        self.mlf = MaskedLocalFusion(
            in_channels=dim,
            roi_output_size=7,
            sampling_ratio=2,
        )

        # ── ImageNet normalisation stats ──────────────────────────────────
        self.register_buffer('_mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ── Load pretrained weights ───────────────────────────────────────
        if ckpt_path is not None:
            self._load_pretrained_weights(ckpt_path)

        # ── Freeze strategy ───────────────────────────────────────────────
        self._apply_freeze_strategy()

    # ── Freeze strategy ───────────────────────────────────────────────────

    def _apply_freeze_strategy(self):
        if not self.use_mlf:
            for p in self.parameters():
                p.requires_grad = False
        else:
            # Freeze DINOv2 + DPT head
            for p in self.pretrained.parameters():
                p.requires_grad = False
            for p in self.depth_head.parameters():
                p.requires_grad = False   # ← freeze hẳn, không cần lr=0 trick

            # Chỉ train MLF
            for p in self.mlf.parameters():
                p.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.parameters())
            Log.info(f"Trainable: {trainable:,} / {total:,} params (MLF only)")

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        encoder: str = 'vitl',
        use_mlf: bool = True,
        **hf_kwargs,
    ) -> "PromptDA":
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = cls.HF_REPOS[encoder]

        if Path(pretrained_model_name_or_path).exists():
            ckpt_path = pretrained_model_name_or_path
        else:
            ckpt_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.ckpt",
                **hf_kwargs,
            )

        return cls(encoder=encoder, ckpt_path=ckpt_path, use_mlf=use_mlf)

    # ── Checkpoint loading ────────────────────────────────────────────────

    def _load_pretrained_weights(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            Log.warn(f"Checkpoint does not exist: {ckpt_path}")
            return

        Log.info(f'Loading checkpoint: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # ── Identical to original: expects 'state_dict' with 'model.' prefix
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Strip 9-char prefix 'model.xxx' exactly as original load_checkpoint does
        # but fall back to generic stripping for flexibility
        first_key = next(iter(state_dict))
        if first_key.startswith('model.'):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            Log.info("Stripped prefix: 'model.'")
        elif '.' in first_key:
            prefix = first_key.split('.')[0] + '.'
            if all(k.startswith(prefix) for k in state_dict):
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
                Log.info(f"Stripped prefix: '{prefix}'")

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        mlf_missing      = [k for k in missing if k.startswith('mlf')]
        critical_missing = [k for k in missing if not k.startswith('mlf')]

        if critical_missing:
            Log.warn(f'Missing keys (unexpected): {critical_missing}')
        if unexpected:
            Log.warn(f'Unexpected keys: {unexpected}')
        Log.info(f'MLF keys random init (expected): {len(mlf_missing)}')

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        prompt_depth: torch.Tensor,
        boxes: Optional[list[torch.Tensor]] = None,
        return_intermediate: bool = False,
    ):
        assert prompt_depth is not None, 'prompt_depth is required'

        prompt_depth, min_val, max_val = self.normalize(prompt_depth)

        h, w    = x.shape[-2:]
        patch_h = h // self.patch_size
        patch_w = w // self.patch_size

        # ── Backbone forward (identical to original) ──────────────────────
        features = self.pretrained.get_intermediate_layers(
            (x - self._mean) / self._std,
            self.model_config['layer_idxs'],
            return_class_token=True,
        )

        if self.use_mlf and boxes is not None:
            # Convert tuple→list so we can splice the deepest feature map
            features = list(features)

            deepest_tokens, cls_token = features[-1]   # (B, N, C)
            f_global = deepest_tokens.permute(0, 2, 1).reshape(
                deepest_tokens.shape[0],
                deepest_tokens.shape[-1],
                patch_h, patch_w,
            )                                           # (B, C, pH, pW)

            f_enhanced    = self.mlf(f_global, boxes)  # (B, C, pH, pW)
            enhanced_tokens = f_enhanced.flatten(2).permute(0, 2, 1).contiguous()
            features[-1]  = (enhanced_tokens, cls_token)

        # ── DPT head (identical call signature to original) ───────────────
        depth = self.depth_head(features, patch_h, patch_w, prompt_depth)
        depth = self.denormalize(depth, min_val, max_val)

        if return_intermediate and self.use_mlf and boxes is not None:
            return depth, f_enhanced
        return depth

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        prompt_depth: torch.Tensor,
        boxes: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.forward(image, prompt_depth, boxes=boxes)

    # ── Helpers ───────────────────────────────────────────────────────────

    @property
    def model_config(self):
        return model_configs[self.encoder]

    def normalize(self, prompt_depth: torch.Tensor):
        """Identical to original normalize()."""
        B = prompt_depth.shape[0]
        min_val = torch.quantile(
            prompt_depth.reshape(B, -1), 0., dim=1, keepdim=True
        )[:, :, None, None]
        max_val = torch.quantile(
            prompt_depth.reshape(B, -1), 1., dim=1, keepdim=True
        )[:, :, None, None]
        prompt_depth = (prompt_depth - min_val) / (max_val - min_val)
        return prompt_depth, min_val, max_val

    def denormalize(self, depth, min_val, max_val):
        """Identical to original denormalize()."""
        return depth * (max_val - min_val) + min_val