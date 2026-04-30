"""
promptda/promptda.py

Main PromptDA model class.

Two comparison modes:

  Baseline (use_mlf=False):
        - Load full PromptDA pretrained weights (DINOv2 + DPT head)
        - Freeze the whole model
        - Run zero-shot evaluation on ARKitScenes

  Experiment (use_mlf=True):
        - Load full PromptDA pretrained weights (DINOv2 + DPT head)
        - Freeze DINOv2
        - Train only the MLF projector (random init) on ARKitScenes
        - Evaluate after training
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
        from_scratch: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_mlf = use_mlf
        self.from_scratch = from_scratch
        model_config = model_configs[encoder]

        # Backbone: DINOv2.
        module_path      = Path(__file__)
        package_base_dir = str(Path(*module_path.parts[:-2]))

        # Load DINOv2 backbone (pretrained or from scratch)
        if from_scratch:
            # Initialize DINOv2 from scratch without pretrained weights
            self.pretrained  = torch.hub.load(
                f'{package_base_dir}/torchhub/facebookresearch_dinov2_main',
                'dinov2_{:}14'.format(encoder),
                source='local',
                pretrained=False,
            )
        else:
            # Load pretrained DINOv2
            self.pretrained  = torch.hub.load(
                f'{package_base_dir}/torchhub/facebookresearch_dinov2_main',
                'dinov2_{:}14'.format(encoder),
                source='local',
                pretrained=True,
            )

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        # Decoder: DPT head.
        self.depth_head = DPTHead(
            nclass=1,
            in_channels=dim,
            features=model_config['features'],
            out_channels=model_config['out_channels'],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act=self.output_act,
        )

        # MLF module: only initialize if use_mlf is True
        if self.use_mlf:
            self.mlf = MaskedLocalFusion(
                in_channels=dim,
                roi_output_size=7,
                sampling_ratio=2,
            )

        # ImageNet normalization stats.
        self.register_buffer('_mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Load pretrained weights if provided and not training from scratch
        if ckpt_path is not None and not from_scratch:
            self._load_pretrained_weights(ckpt_path)

        # Apply mode-specific freezing strategy
        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self):
        """
        Apply freezing strategy based on training mode:

        - `from_scratch=True`: All parameters trainable (DINOv2 + DPT)
        - `from_scratch=False, use_mlf=False`: Freeze all parameters (baseline zero-shot)
        - `from_scratch=False, use_mlf=True`: Freeze DINOv2, train DPT + MLF
        """
        if self.from_scratch:
            # Train all parameters from scratch
            for p in self.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.parameters())
            Log.info(
                f"Mode: FROM_SCRATCH — All parameters trainable ({trainable:,} / {total:,} params)"
            )
        elif not self.use_mlf:
            # Freeze all parameters (baseline zero-shot)
            for p in self.parameters():
                p.requires_grad = False
            Log.info("Mode: BASELINE — all parameters frozen, zero-shot evaluation")
        else:
            # Freeze backbone only, train DPT + MLF
            for p in self.pretrained.parameters():
                p.requires_grad = False
            # Keep DPT trainable for gradient flow; optimizer can keep lr=0.
            for p in self.depth_head.parameters():
                p.requires_grad = True
            # MLF is trainable.
            for p in self.mlf.parameters():
                p.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.parameters())
            Log.info(
                f"Mode: EXPERIMENT — DINOv2 frozen | "
                f"DPT Head + MLF trainable ({trainable:,} / {total:,} params)"
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        encoder: str = 'vitl',
        use_mlf: bool = True,
        from_scratch: bool = False,
        **hf_kwargs,
    ) -> "PromptDA":
        """
        Load PromptDA from Hugging Face Hub or a local path.

        Args:
            pretrained_model_name_or_path:
                None       → use default repo from `HF_REPOS[encoder]`
                Local path → load directly
                HF repo id → download to cache
            encoder : 'vits' | 'vitb' | 'vitl'
            use_mlf : False = baseline, True = experiment
            from_scratch : True = train from scratch, False = load pretrained weights
        """
        ckpt_path = None
        if not from_scratch:
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

        return cls(encoder=encoder, ckpt_path=ckpt_path, use_mlf=use_mlf, from_scratch=from_scratch)

    def _load_pretrained_weights(self, ckpt_path: str):
        """
        Load checkpoint with automatic prefix stripping and `strict=False`.
        Missing MLF keys are expected and remain randomly initialized.
        """
        if not os.path.exists(ckpt_path):
            Log.warn(f"Checkpoint does not exist: {ckpt_path}")
            return

        Log.info(f'Loading checkpoint: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Support Lightning format, trainer format, and raw state-dict format.
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Strip common module prefix automatically.
        first_key = next(iter(state_dict))
        if '.' in first_key:
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

    def forward(
        self,
        image: torch.Tensor,
        prompt_depth: torch.Tensor,
        boxes: Optional[list[torch.Tensor]] = None,
        return_intermediate: bool = False,
    ):
        assert prompt_depth is not None, 'prompt_depth is required'

        prompt_depth, min_val, max_val = self._normalize_prompt(prompt_depth)

        h, w    = image.shape[-2:]
        patch_h = h // self.patch_size
        patch_w = w // self.patch_size

        # Get backbone features
        if self.from_scratch:
            # Training from scratch - keep gradients
            features = list(self.pretrained.get_intermediate_layers(
                (image - self._mean) / self._std,
                self.model_config['layer_idxs'],
                return_class_token=True,
            ))
        else:
            # Using pretrained backbone - run under no_grad
            with torch.no_grad():
                features = list(self.pretrained.get_intermediate_layers(
                    (image - self._mean) / self._std,
                    self.model_config['layer_idxs'],
                    return_class_token=True,
                ))

        # Use deepest feature map
        deepest_tokens, cls_token = features[-1]
        f_global = deepest_tokens.permute(0, 2, 1).reshape(
            deepest_tokens.shape[0],
            deepest_tokens.shape[-1],
            patch_h, patch_w,
        )

        # Apply MLF if enabled and boxes are provided
        if self.use_mlf and boxes is not None:
            if self.from_scratch:
                # Training from scratch - keep gradients through MLF
                f_enhanced = self.mlf(f_global, boxes)
            else:
                # Using pretrained backbone - detach global features
                f_global_detached = f_global.detach()
                f_enhanced = self.mlf(f_global_detached, boxes)
        else:
            f_enhanced = f_global

        # Replace deepest token map with enhanced tokens
        enhanced_tokens = f_enhanced.flatten(2).permute(0, 2, 1).contiguous()

        # Handle shallow features
        if self.from_scratch:
            # Training from scratch - keep all gradients
            features_for_head = features[:-1] + [(enhanced_tokens, cls_token)]
        else:
            # Using pretrained backbone - keep shallow features frozen
            with torch.no_grad():
                frozen_features = features[:-1]
            features_for_head = frozen_features + [(enhanced_tokens, cls_token.detach())]

        # DPT head consumes enhanced tokens to produce final depth
        depth = self.depth_head(features_for_head, patch_h, patch_w, prompt_depth)
        depth = self._denormalize_depth(depth, min_val, max_val)

        if return_intermediate:
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

    @property
    def model_config(self):
        return model_configs[self.encoder]

    def _normalize_prompt(self, prompt_depth: torch.Tensor):
        B    = prompt_depth.shape[0]
        flat = prompt_depth.reshape(B, -1)
        min_val = flat.quantile(0., dim=1).view(B, 1, 1, 1)
        max_val = flat.quantile(1., dim=1).view(B, 1, 1, 1)
        normalized = (prompt_depth - min_val) / (max_val - min_val + 1e-8)
        return normalized, min_val, max_val

    def _denormalize_depth(self, depth, min_val, max_val):
        return depth * (max_val - min_val) + min_val