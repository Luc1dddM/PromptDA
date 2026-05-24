from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F

from promptda.model.config import model_configs
from promptda.promptda import PromptDA
from promptda.utils.logger import Log
from training.metrics import compute_edge_strength


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _depthwise_gaussian_init(module: nn.Conv2d):
    kernel_h, kernel_w = module.kernel_size
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=kernel_h),
        torch.linspace(-1.0, 1.0, steps=kernel_w),
        indexing="ij",
    )
    dist2 = xs.square() + ys.square()
    kernel = torch.exp(-dist2 / 0.3)
    kernel = kernel / kernel.sum()
    with torch.no_grad():
        module.weight.zero_()
        module.weight[0, 0] = kernel


@dataclass
class SACGOutput:
    refined_depth: torch.Tensor
    coarse_depth: torch.Tensor
    gate_map: torch.Tensor
    c_grad: torch.Tensor
    f_lidar: torch.Tensor
    delta_depth: torch.Tensor
    edge_strength: torch.Tensor
    dpt_feat: torch.Tensor | None = None


class SACGModule(nn.Module):
    def __init__(
        self,
        rgb_channels: int = 3,
        dpt_channels: int = 0,
        hidden_channels: int = 32,
        lidar_sigma: float = 50.0,
        learnable_lidar: bool = False,
        lidar_kernel_size: int = 31,
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.dpt_channels = dpt_channels
        self.lidar_sigma = lidar_sigma
        self.learnable_lidar = learnable_lidar

        gate_in_channels = 3
        refine_in_channels = rgb_channels + 1 + dpt_channels

        self.gate_net = nn.Sequential(
            _conv_block(gate_in_channels, 16),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

        self.refine_net = nn.Sequential(
            _conv_block(refine_in_channels, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1),
        )

        if self.learnable_lidar:
            self.lidar_filter = nn.Conv2d(
                1,
                1,
                kernel_size=lidar_kernel_size,
                stride=1,
                padding=lidar_kernel_size // 2,
                groups=1,
                bias=False,
            )
            _depthwise_gaussian_init(self.lidar_filter)
        else:
            self.lidar_filter = None

    def compute_gradient_consistency(
        self,
        rgb: torch.Tensor,
        d_coarse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g_rgb = compute_edge_strength(rgb)
        g_depth = compute_edge_strength(d_coarse.repeat(1, 3, 1, 1))
        c_grad = 1.0 - torch.abs(g_rgb - g_depth)
        return c_grad.clamp(0.0, 1.0), g_rgb

    def compute_lidar_field(self, sparse: torch.Tensor) -> torch.Tensor:
        if self.learnable_lidar:
            sparse_mask = (sparse > 0).float()
            field = self.lidar_filter(sparse_mask)
            field = field - field.amin(dim=(-2, -1), keepdim=True)
            denom = field.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            return (field / denom).clamp(0.0, 1.0)

        sparse_np = sparse.detach().cpu().numpy()
        fields = []
        for sample in sparse_np:
            valid_mask = sample[0] > 0
            dist = distance_transform_edt(~valid_mask)
            field = np.exp(-dist / self.lidar_sigma).astype(np.float32)
            fields.append(field[None, ...])
        return torch.from_numpy(np.stack(fields, axis=0)).to(device=sparse.device, dtype=sparse.dtype)

    def forward(
        self,
        rgb: torch.Tensor,
        sparse: torch.Tensor,
        d_coarse: torch.Tensor,
        dpt_feat: Optional[torch.Tensor] = None,
    ) -> SACGOutput:
        c_grad, edge_strength = self.compute_gradient_consistency(rgb, d_coarse)
        f_lidar = self.compute_lidar_field(sparse)

        gate_in = torch.cat([c_grad, f_lidar, d_coarse], dim=1)
        gate_map = torch.sigmoid(self.gate_net(gate_in))

        refine_inputs = [rgb, d_coarse]
        dpt_feat_resized = None
        if dpt_feat is not None:
            dpt_feat_resized = F.interpolate(
                dpt_feat,
                size=rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            refine_inputs.append(dpt_feat_resized)

        ref_in = torch.cat(refine_inputs, dim=1)
        delta_depth = self.refine_net(ref_in)
        refined_depth = d_coarse + gate_map * delta_depth

        return SACGOutput(
            refined_depth=refined_depth,
            coarse_depth=d_coarse,
            gate_map=gate_map,
            c_grad=c_grad,
            f_lidar=f_lidar,
            delta_depth=delta_depth,
            edge_strength=edge_strength,
            dpt_feat=dpt_feat_resized,
        )


class PromptDASACG(nn.Module):
    def __init__(
        self,
        promptda: PromptDA,
        sacg: SACGModule,
        feature_hook_module: str = "scratch.output_conv1",
    ):
        super().__init__()
        self.promptda = promptda
        self.sacg = sacg
        self._feat_cache: dict[str, torch.Tensor | None] = {"dpt_feat": None}
        self._hook_handle = self._register_feature_hook(feature_hook_module)
        self._freeze_promptda()

    def _freeze_promptda(self):
        for p in self.promptda.parameters():
            p.requires_grad = False
        self.promptda.eval()

    def _register_feature_hook(self, module_path: str):
        module = self.promptda.depth_head
        for part in module_path.split("."):
            module = getattr(module, part)

        def _save_output(_module, _inputs, output):
            self._feat_cache["dpt_feat"] = output

        Log.info(f"Registering SACG feature hook on depth_head.{module_path}")
        return module.register_forward_hook(_save_output)

    @classmethod
    def from_pretrained(
        cls,
        encoder: str = "vitl",
        promptda_ckpt: Optional[str] = None,
        dpt_variant: str = "legacy",
        sacg_ckpt: Optional[str] = None,
        learnable_lidar: bool = False,
    ) -> "PromptDASACG":
        promptda = PromptDA.from_pretrained(
            pretrained_model_name_or_path=promptda_ckpt,
            encoder=encoder,
            use_mlf=False,
            dpt_variant=dpt_variant,
        )
        dpt_channels = model_configs[encoder]["features"] // 2
        sacg = SACGModule(
            dpt_channels=dpt_channels,
            learnable_lidar=learnable_lidar,
        )
        model = cls(promptda=promptda, sacg=sacg)
        if sacg_ckpt is not None:
            model.load_sacg_checkpoint(sacg_ckpt)
        return model

    def load_sacg_checkpoint(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if all(key.startswith("sacg.") for key in state_dict.keys()):
            state_dict = {key[len("sacg."):]: value for key, value in state_dict.items()}
        elif all(key.startswith("module.sacg.") for key in state_dict.keys()):
            state_dict = {key[len("module.sacg."):]: value for key, value in state_dict.items()}
        missing, unexpected = self.sacg.load_state_dict(state_dict, strict=False)
        if missing:
            Log.warn(f"SACG missing keys: {missing}")
        if unexpected:
            Log.warn(f"SACG unexpected keys: {unexpected}")
        Log.info(f"Loaded SACG checkpoint: {ckpt_path}")

    def forward(self, rgb: torch.Tensor, sparse_depth: torch.Tensor) -> dict[str, torch.Tensor]:
        self._feat_cache["dpt_feat"] = None
        with torch.no_grad():
            d_coarse = self.promptda(rgb, sparse_depth)
        sacg_output = self.sacg(
            rgb=rgb,
            sparse=sparse_depth,
            d_coarse=d_coarse,
            dpt_feat=self._feat_cache.get("dpt_feat"),
        )
        return {
            "refined_depth": sacg_output.refined_depth,
            "coarse_depth": sacg_output.coarse_depth,
            "gate_map": sacg_output.gate_map,
            "c_grad": sacg_output.c_grad,
            "f_lidar": sacg_output.f_lidar,
            "delta_depth": sacg_output.delta_depth,
            "edge_strength": sacg_output.edge_strength,
            "dpt_feat": sacg_output.dpt_feat,
        }


class SACGLoss(nn.Module):
    def __init__(
        self,
        boundary_weight: float = 0.5,
        gate_weight: float = 0.1,
        gate_warmup_epoch: int = 5,
        use_boundary_loss: bool = True,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.gate_weight = gate_weight
        self.gate_warmup_epoch = gate_warmup_epoch
        self.use_boundary_loss = use_boundary_loss

    def forward(
        self,
        refined_depth: torch.Tensor,
        coarse_depth: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        gate_map: torch.Tensor,
        c_grad: torch.Tensor,
        edge_strength: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if valid_mask.any():
            l_main = F.l1_loss(refined_depth[valid_mask], target[valid_mask])
        else:
            l_main = refined_depth.new_tensor(0.0)

        if self.use_boundary_loss:
            w_edge = torch.exp(10.0 * edge_strength).detach()
            l_boundary = (w_edge * torch.abs(refined_depth - target) * valid_mask.float()).mean()
        else:
            l_boundary = refined_depth.new_tensor(0.0)

        l_gate = (gate_map * c_grad.detach()).mean()

        total = l_main
        if self.use_boundary_loss:
            total = total + self.boundary_weight * l_boundary
        if epoch >= self.gate_warmup_epoch:
            total = total + self.gate_weight * l_gate

        return total, {
            "L_main": float(l_main.item()),
            "L_boundary": float(l_boundary.item()),
            "L_gate": float(l_gate.item()),
            "loss_total": float(total.item()),
        }
