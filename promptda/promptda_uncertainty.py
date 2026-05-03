"""
PromptDA Uncertainty variant — 2-channel head for aleatoric uncertainty.

Extends PromptDA to output both depth (μ) and log-uncertainty (s = log σ).
Channel 0: μ (depth, with sigmoid activation)
Channel 1: s (log σ, unbounded — clamped in loss for numerical safety)

Strategy A weight loading:
    Pass `promptda_ckpt_path` to load the full PromptDA checkpoint.
    output_conv2 is skipped automatically (1ch → 2ch shape mismatch, strict=False).
    All other DPT head weights (scratch, reassemble) are loaded and used as init.

Usage:
    # Strategy A (recommended): load full PromptDA checkpoint
    model = PromptDAUncertainty(
        encoder='vitl',
        dpt_variant='skip_concat_1x1',
        promptda_ckpt_path='/path/to/promptda.pth',
    )

    # Inference only (load own trained checkpoint)
    model = PromptDAUncertainty(
        encoder='vitl',
        ckpt_path='/path/to/uncertainty_ckpt.pth',
    )

    output = model(image, prompt_depth)   # dict: {"mu": [B,1,H,W], "s": [B,1,H,W]}
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from promptda.model.config import model_configs
from promptda.model.dpt import DPTHead
from promptda.utils.logger import Log


class PromptDAUncertainty(nn.Module):
    """PromptDA with 2-channel uncertainty head (μ depth + log σ).

    Backbone (DINOv2) is frozen; only the DPT decoder head is trained.
    The head produces 2 channels:
        out[:, 0] → μ (depth in [0,1] after sigmoid)
        out[:, 1] → s = log(σ) (raw log-uncertainty)

    Weight initialisation (Strategy A):
        1. DINOv2 ViT: loaded from torch hub (pretrained).
        2. DPT head (scratch + reassemble): loaded from PromptDA checkpoint.
        3. output_conv2: randomly initialised (shape changes 1ch → 2ch).
    """

    patch_size = 14
    use_bn = False
    use_clstoken = False
    # kept for config compatibility; actual sigmoid applied manually to channel 0
    output_act = 'sigmoid'

    def __init__(
        self,
        encoder: str = 'vitl',
        ckpt_path: str = None,
        dpt_variant: str = 'legacy',
        promptda_ckpt_path: str = None,
    ):
        """
        Args:
            encoder: ViT encoder size — 'vits' | 'vitb' | 'vitl'.
            ckpt_path: Path to a previously saved PromptDAUncertainty checkpoint
                       (used to resume training or for inference). Applied AFTER
                       promptda_ckpt_path so it always wins.
            dpt_variant: DPT head variant — must match the variant used to train
                         the PromptDA checkpoint supplied via promptda_ckpt_path.
            promptda_ckpt_path: Path to a pretrained PromptDA (1-channel) checkpoint.
                                Implements Strategy A: loads all weights except
                                output_conv2 (skipped due to channel mismatch).
        """
        super().__init__()
        model_config = model_configs[encoder]

        self.encoder = encoder
        self.model_config = model_config
        self.dpt_variant = dpt_variant
        self.nclass = 2  # 2 channels: mu + log(sigma)

        # ------------------------------------------------------------------ #
        # 1. DINOv2 backbone (ViT)
        # ------------------------------------------------------------------ #
        module_path = Path(__file__)
        package_base_dir = str(Path(*module_path.parts[:-2]))
        self.pretrained = torch.hub.load(
            f'{package_base_dir}/torchhub/facebookresearch_dinov2_main',
            'dinov2_{:}14'.format(encoder),
            source='local',
            pretrained=True,
        )
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        # ------------------------------------------------------------------ #
        # 2. DPT decoder head (2-channel output)
        # ------------------------------------------------------------------ #
        self.depth_head = DPTHead(
            nclass=self.nclass,
            in_channels=dim,
            features=model_config['features'],
            out_channels=model_config['out_channels'],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act=self.output_act,
            dpt_variant=self.dpt_variant,
        )

        # Freeze backbone; DPT head is fully trainable
        self._freeze_backbone_only_train_head()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        Log.info(
            f"Trainable: {trainable:,} / {total:,} params "
            f"(DPT head only, 2-ch uncertainty)"
        )

        # ------------------------------------------------------------------ #
        # 3. DINOv2 normalisation stats
        # ------------------------------------------------------------------ #
        self.register_buffer(
            '_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            '_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # ------------------------------------------------------------------ #
        # 4. Weight loading order:
        #    a) PromptDA checkpoint (Strategy A) — loads scratch + reassemble
        #    b) PromptDAUncertainty checkpoint   — resumes own training
        #    b always wins over a if both are supplied.
        # ------------------------------------------------------------------ #
        if promptda_ckpt_path is not None:
            self._load_promptda_weights(promptda_ckpt_path)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    # ---------------------------------------------------------------------- #
    # Class-method constructors
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = None,
        encoder: str = 'vitl',
        dpt_variant: str = 'legacy',
        **hf_kwargs,
    ):
        if hf_kwargs:
            Log.warn(f"Unused kwargs in from_pretrained: {list(hf_kwargs.keys())}")
        if pretrained_model_name_or_path is not None:
            Log.warn(
                "Ignoring pretrained_model_name_or_path. "
                "Only DINOv2 backbone pretrained weights are loaded. "
                "Use promptda_ckpt_path=... to load PromptDA weights (Strategy A)."
            )
        return cls(encoder=encoder, ckpt_path=None, dpt_variant=dpt_variant)

    # ---------------------------------------------------------------------- #
    # Weight loading helpers
    # ---------------------------------------------------------------------- #

    def _freeze_backbone_only_train_head(self):
        for p in self.pretrained.parameters():
            p.requires_grad = False
        for p in self.depth_head.parameters():
            p.requires_grad = True

    def _load_promptda_weights(self, promptda_ckpt_path: str):
        """Strategy A: load PromptDA (1-ch) checkpoint into 2-ch model.

        Keys that belong to output_conv2 are skipped automatically because
        their shapes differ (out_channels=1 vs out_channels=2).  Every other
        key in depth_head (scratch layers, reassemble blocks) is loaded and
        provides a warm start for the uncertainty head.

        Args:
            promptda_ckpt_path: Path to PromptDA .pth checkpoint file.
        """
        if not os.path.exists(promptda_ckpt_path):
            Log.warn(
                f"PromptDA checkpoint not found: {promptda_ckpt_path}. "
                "Training from random DPT head init (Strategy C)."
            )
            return

        Log.info(f"[Strategy A] Loading PromptDA weights from: {promptda_ckpt_path}")
        ckpt = torch.load(promptda_ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip common 'model.' prefix if present (e.g. lightning checkpoints)
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items()}

        # strict=False: output_conv2 keys are silently skipped (shape mismatch)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        # Separate expected mismatches (output_conv2) from real problems
        output_conv2_keys = [k for k in missing if 'output_conv2' in k]
        real_missing = [k for k in missing if 'output_conv2' not in k]

        Log.info(
            f"[Strategy A] Loaded successfully. "
            f"Reinitialised (output_conv2, 1ch→2ch): {output_conv2_keys}"
        )
        if real_missing:
            Log.warn(
                f"[Strategy A] Unexpected missing keys — check that --dpt_variant "
                f"matches the PromptDA checkpoint: {real_missing}"
            )
        if unexpected:
            Log.warn(
                f"[Strategy A] Keys in checkpoint not present in model: {unexpected}"
            )

    def load_checkpoint(self, ckpt_path: str):
        """Load a PromptDAUncertainty checkpoint (2-channel, own training)."""
        if not os.path.exists(ckpt_path):
            Log.warn(f"Checkpoint not found: {ckpt_path}")
            return

        Log.info(f"Loading PromptDAUncertainty checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            Log.warn(f"Missing keys: {missing}")
        if unexpected:
            Log.warn(f"Unexpected keys: {unexpected}")

    def trainable_parameters(self):
        return [p for p in self.depth_head.parameters() if p.requires_grad]

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor, prompt_depth: torch.Tensor = None):
        """Forward pass returning mu (depth) and s (log-sigma).

        Args:
            x: RGB image tensor [B, 3, H, W], values in [0, 1].
            prompt_depth: Sparse LiDAR depth prompt [B, 1, H, W].

        Returns:
            dict with keys:
                "mu":     [B, 1, H, W] predicted depth in original metric scale.
                "s":      [B, 1, H, W] log(sigma) — raw uncertainty logits.
                          Use exp(s) to get sigma; clamp s before exp in loss.
                "raw_mu": [B, 1, H, W] depth after sigmoid, before denorm
                          (useful for debugging normalisation).
        """
        assert prompt_depth is not None, 'prompt_depth is required'

        prompt_depth_norm, min_val, max_val = self.normalize(prompt_depth)

        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(
            (x - self._mean) / self._std,
            self.model_config['layer_idxs'],
            return_class_token=True,
        )

        patch_h = h // self.patch_size
        patch_w = w // self.patch_size

        # DPT head outputs [B, 2, H, W] — both channels in normalised space
        raw_out = self.depth_head(features, patch_h, patch_w, prompt_depth_norm)

        # Split channels
        mu_norm = raw_out[:, 0:1, :, :]  # depth (normalised)
        s_log   = raw_out[:, 1:2, :, :]  # log(sigma) — no activation

        # Sigmoid only on depth channel → [0, 1]
        mu_norm = torch.sigmoid(mu_norm)

        # Denormalise depth back to metric scale
        mu = self.denormalize(mu_norm, min_val, max_val)

        return {
            "mu":     mu,
            "s":      s_log,
            "raw_mu": mu_norm,
        }

    # ---------------------------------------------------------------------- #
    # Inference helper
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(self, image: torch.Tensor, prompt_depth: torch.Tensor) -> torch.Tensor:
        """Inference shortcut — returns depth map only (no uncertainty)."""
        return self.forward(image, prompt_depth)["mu"]

    # ---------------------------------------------------------------------- #
    # Depth normalisation (per-sample, min-max)
    # ---------------------------------------------------------------------- #

    def normalize(self, prompt_depth: torch.Tensor):
        """Per-sample min-max normalisation of the LiDAR prompt.

        Returns:
            normalised prompt, min_val [B,1,1,1], max_val [B,1,1,1]
        """
        B = prompt_depth.shape[0]
        flat = prompt_depth.reshape(B, -1)
        min_val = flat.min(dim=1).values[:, None, None, None]
        max_val = flat.max(dim=1).values[:, None, None, None]
        prompt_norm = (prompt_depth - min_val) / (max_val - min_val + 1e-8)
        return prompt_norm, min_val, max_val

    def denormalize(
        self,
        depth: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> torch.Tensor:
        return depth * (max_val - min_val) + min_val