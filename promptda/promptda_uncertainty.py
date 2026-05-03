"""
PromptDA Uncertainty variant — 2-channel head for aleatoric uncertainty.

Extends PromptDA to output both depth (μ) and log-uncertainty (s = log σ).
Channel 0: μ (depth, with sigmoid activation)
Channel 1: s (log σ, unbounded — clamped in loss for numerical safety)

Strategy A weight loading:
    Pass `promptda_ckpt_path` as either:
      - A local file path  → loaded directly.
      - A HuggingFace repo ID (e.g. "depth-anything/prompt-depth-anything-vitl")
        → downloaded via hf_hub_download.
    Handles both 'model.' and 'pipeline.' key prefixes from HF checkpoints.
    Shape-mismatched keys (output_conv2 final layer: 1ch → 2ch) are removed
    manually before load_state_dict (strict=False does NOT skip shape mismatches).
    Channel 0 of the new 2-ch output_conv2 is warm-started from the pretrained weight.
    All other DPT head weights (scratch, reassemble) are loaded as warm-start.

Usage:
    # Strategy A — from HuggingFace Hub (recommended)
    model = PromptDAUncertainty.from_pretrained(
        encoder='vitl',
        dpt_variant='legacy',
    )

    # Strategy A — from local checkpoint
    model = PromptDAUncertainty.from_pretrained(
        pretrained_model_name_or_path='/path/to/promptda.ckpt',
        encoder='vitl',
    )

    # Resume own uncertainty training
    model = PromptDAUncertainty(
        encoder='vitl',
        ckpt_path='/path/to/uncertainty_ckpt.pth',
    )

    output = model(image, prompt_depth)
    # → dict: {"mu": [B,1,H,W], "s": [B,1,H,W], "raw_mu": [B,1,H,W]}
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download

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
        1. DINOv2 ViT     : loaded from torch hub (pretrained).
        2. DPT head       : loaded from PromptDA checkpoint (local or HF Hub).
        3. output_conv2   : randomly initialised (shape changes 1ch → 2ch).
    """

    patch_size = 14
    use_bn = False
    use_clstoken = False
    output_act = 'sigmoid'  # kept for config compat; sigmoid applied manually to ch0

    # Map encoder → default HuggingFace repo
    HF_REPOS = {
        'vits': 'depth-anything/prompt-depth-anything-vits',
        'vitb': 'depth-anything/prompt-depth-anything-vitb',
        'vitl': 'depth-anything/prompt-depth-anything-vitl',
    }

    def __init__(
        self,
        encoder: str = 'vitl',
        ckpt_path: Optional[str] = None,
        dpt_variant: str = 'legacy',
        promptda_ckpt_path: Optional[str] = None,
    ):
        """
        Args:
            encoder: ViT encoder size — 'vits' | 'vitb' | 'vitl'.
            ckpt_path: Path to a previously saved PromptDAUncertainty checkpoint
                       (resume training or inference). Applied AFTER
                       promptda_ckpt_path so it always wins.
            dpt_variant: DPT head variant — must match the variant used when
                         training the PromptDA checkpoint in promptda_ckpt_path.
            promptda_ckpt_path: Already-resolved local path to a pretrained
                                PromptDA (1-channel) checkpoint.
                                Prefer using from_pretrained() which handles
                                local-vs-HF resolution for you.
        """
        super().__init__()
        model_config = model_configs[encoder]

        self.encoder = encoder
        self.model_config = model_config
        self.dpt_variant = dpt_variant
        self.nclass = 2  # channels: mu + log(sigma)

        # ------------------------------------------------------------------ #
        # 1. DINOv2 backbone (ViT) — always pretrained
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
        #    a) PromptDA checkpoint (Strategy A) — warms up DPT head
        #    b) PromptDAUncertainty checkpoint   — resumes own training
        #    (b) always wins over (a) when both are supplied.
        # ------------------------------------------------------------------ #
        if promptda_ckpt_path is not None:
            self._load_promptda_weights(promptda_ckpt_path)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    # ---------------------------------------------------------------------- #
    # Class-method constructor  (mirrors PromptDA.from_pretrained exactly)
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        encoder: str = 'vitl',
        dpt_variant: str = 'legacy',
        **hf_kwargs,
    ) -> "PromptDAUncertainty":
        """Create a PromptDAUncertainty model with Strategy A weight loading.

        Mirrors the PromptDA.from_pretrained() interface:
          - None            → uses HF_REPOS[encoder] default repo.
          - local path      → loaded directly.
          - HF repo ID      → downloaded via hf_hub_download.

        Args:
            pretrained_model_name_or_path: Local path or HF repo ID.
            encoder: 'vits' | 'vitb' | 'vitl'.
            dpt_variant: Must match the checkpoint architecture.
            **hf_kwargs: Forwarded to hf_hub_download (token=, cache_dir=, ...).

        Returns:
            PromptDAUncertainty with warm-started DPT head (Strategy A).
        """
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = cls.HF_REPOS[encoder]

        if Path(pretrained_model_name_or_path).exists():
            ckpt_path = pretrained_model_name_or_path
            Log.info(f"[from_pretrained] Using local checkpoint: {ckpt_path}")
        else:
            Log.info(
                f"[from_pretrained] Downloading PromptDA checkpoint "
                f"from HF repo: {pretrained_model_name_or_path}"
            )
            ckpt_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.ckpt",
                **hf_kwargs,
            )

        return cls(
            encoder=encoder,
            ckpt_path=None,
            dpt_variant=dpt_variant,
            promptda_ckpt_path=ckpt_path,
        )

    # ---------------------------------------------------------------------- #
    # Weight loading helpers
    # ---------------------------------------------------------------------- #

    def _freeze_backbone_only_train_head(self):
        for p in self.pretrained.parameters():
            p.requires_grad = False
        for p in self.depth_head.parameters():
            p.requires_grad = True

    @staticmethod
    def _strip_prefix(state_dict: dict) -> dict:
        """Strip known wrapper prefixes from checkpoint state_dict keys.

        The official PromptDA HF checkpoint wraps everything under 'pipeline.',
        while Lightning checkpoints use 'model.'.  This method strips either
        prefix so that keys align with our module attribute names.

        Priority: pipeline. > model. > no prefix.
        """
        # Check pipeline. prefix first (PromptDA HF checkpoint)
        if all(k.startswith('pipeline.') for k in state_dict.keys()):
            Log.info("[Strategy A] Stripping 'pipeline.' prefix from checkpoint keys.")
            return {k[len('pipeline.'):]: v for k, v in state_dict.items()}

        # Check model. prefix (Lightning / other wrappers)
        if all(k.startswith('model.') for k in state_dict.keys()):
            Log.info("[Strategy A] Stripping 'model.' prefix from checkpoint keys.")
            return {k[len('model.'):]: v for k, v in state_dict.items()}

        return state_dict

    def _load_promptda_weights(self, promptda_ckpt_path: str):
        """Strategy A: load PromptDA (1-ch) weights into this 2-ch model.

        Handles both 'pipeline.' and 'model.' key prefixes automatically.

        PyTorch's ``load_state_dict(strict=False)`` does **not** silently skip
        shape-mismatched keys — it raises ``RuntimeError``.  We therefore
        manually remove mismatched keys (output_conv2 final layer: 1ch → 2ch)
        before calling ``load_state_dict``, and then warm-start channel 0 of
        the new 2-channel output_conv2 from the pretrained 1-channel weight.

        Args:
            promptda_ckpt_path: Absolute local path to the PromptDA checkpoint.
        """
        Log.info(f"[Strategy A] Loading PromptDA weights from: {promptda_ckpt_path}")
        ckpt = torch.load(promptda_ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip wrapper prefix so keys match our attribute names
        state_dict = self._strip_prefix(state_dict)

        # ── Filter out shape-mismatched keys ──────────────────────────────
        # strict=False does NOT skip shape mismatches (raises RuntimeError).
        # We must remove them manually before load_state_dict.
        model_sd = self.state_dict()
        mismatched_keys = {}  # key → ckpt tensor (saved for warm-start)
        filtered_sd = {}
        for k, v in state_dict.items():
            if k in model_sd and v.shape != model_sd[k].shape:
                mismatched_keys[k] = v
                Log.info(
                    f"[Strategy A] Shape mismatch — skipping '{k}': "
                    f"ckpt {list(v.shape)} vs model {list(model_sd[k].shape)}"
                )
            else:
                filtered_sd[k] = v

        missing, unexpected = self.load_state_dict(filtered_sd, strict=False)

        # ── Warm-start output_conv2 channel 0 from pretrained 1-ch weight ─
        # This gives the depth (μ) channel a much better init than random.
        for k, ckpt_tensor in mismatched_keys.items():
            if 'output_conv2' not in k:
                continue
            param = model_sd[k]
            if ckpt_tensor.dim() >= 1 and param.dim() >= 1:
                # Weight: ckpt [1, C_in, ...] → model [2, C_in, ...]
                # Bias:   ckpt [1]            → model [2]
                # Copy pretrained into channel 0; channel 1 stays random init.
                with torch.no_grad():
                    n_copy = min(ckpt_tensor.shape[0], param.shape[0])
                    param_ref = dict(self.named_parameters()).get(k)
                    if param_ref is not None:
                        param_ref.data[:n_copy] = ckpt_tensor[:n_copy]
                        Log.info(
                            f"[Strategy A] Warm-started '{k}' channel 0 "
                            f"from pretrained 1-ch weight."
                        )

        # ── Classify remaining missing keys ───────────────────────────────
        # Missing keys that were intentionally skipped (shape mismatch) are
        # NOT in the `missing` list — they were removed from filtered_sd.
        real_missing = [k for k in missing if k not in mismatched_keys]

        Log.info(
            f"[Strategy A] Loaded successfully. "
            f"Shape-mismatched keys (warm-started ch0): {list(mismatched_keys.keys())}"
        )
        if real_missing:
            Log.warn(
                f"[Strategy A] Unexpected missing keys — verify that --dpt_variant "
                f"matches the PromptDA checkpoint architecture: {real_missing}"
            )
        if unexpected:
            Log.warn(
                f"[Strategy A] Keys in checkpoint not found in model "
                f"(safe to ignore): {unexpected}"
            )

    def load_checkpoint(self, ckpt_path: str):
        """Load a PromptDAUncertainty checkpoint (2-channel, own training)."""
        if not os.path.exists(ckpt_path):
            Log.warn(f"Checkpoint not found: {ckpt_path}")
            return

        Log.info(f"Loading PromptDAUncertainty checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = self._strip_prefix(state_dict)

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
                          Use exp(s) to get sigma; clamp s before exp in the loss.
                "raw_mu": [B, 1, H, W] depth after sigmoid, before denorm.
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

        # DPT head outputs [B, 2, H, W]
        raw_out = self.depth_head(features, patch_h, patch_w, prompt_depth_norm)

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
            (normalised_prompt, min_val [B,1,1,1], max_val [B,1,1,1])
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