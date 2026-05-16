# Copyright (c) 2024, Depth Anything V2
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from promptda.model.blocks import _make_scratch, _make_fusion_block


class CrossAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, query_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = query_map.shape
        if kv_map.shape[-2:] != (h, w):
            kv_map = F.interpolate(kv_map, size=(h, w), mode="bilinear", align_corners=False)

        q = query_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        kv = kv_map.permute(0, 2, 3, 1).reshape(b, h * w, c)

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(query=qn, key=kvn, value=kvn, need_weights=False)
        fused = q + attn_out
        return fused.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def _pick_num_heads(channels: int) -> int:
    if channels % 8 == 0:
        return 8
    if channels % 4 == 0:
        return 4
    if channels % 2 == 0:
        return 2
    return 1


MAX_ATTENTION_TOKENS = 4096

def _maybe_pool_for_attention(x: torch.Tensor, max_tokens: int = MAX_ATTENTION_TOKENS) -> torch.Tensor:
    h, w = x.shape[-2:]
    tokens = h * w
    if tokens <= max_tokens:
        return x
    scale = (tokens / max_tokens) ** 0.5
    new_h = max(1, int(h / scale))
    new_w = max(1, int(w / scale))
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _cross_attend_with_cap(attn: CrossAttention2D, query_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
    kv_small = _maybe_pool_for_attention(kv_map)
    return attn(query_map, kv_small)



class DPTHead(nn.Module):
    def __init__(self,
                 nclass,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid',
                 dpt_variant='legacy'):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.dpt_variant = dpt_variant
        if isinstance(in_channels, int):
            in_channels = [in_channels] * len(out_channels)
            self._expects_tokens = True
        else:
            in_channels = list(in_channels)
            self._expects_tokens = False
        self.in_channels = in_channels
        if len(in_channels) != len(out_channels):
            raise ValueError(
                f"Expected {len(out_channels)} input feature channels, got {len(in_channels)}."
            )

        if self.dpt_variant not in {'legacy', 'skip_concat_1x1', 'hybrid_ca_shallow_concat'}:
            raise ValueError(f"Unsupported dpt_variant: {self.dpt_variant}")

        self.projects = nn.ModuleList([
            nn.Conv2d(c, out_c, kernel_size=1)
            for c, out_c in zip(in_channels, out_channels)
        ])

        if self._expects_tokens:
            self.resize_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1)
            ])
        else:
            self.resize_layers = nn.ModuleList([
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
            ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for c in in_channels:
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * c, c),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(
            features, use_bn)

        if self.dpt_variant == 'skip_concat_1x1':
            self.skip_fuse3 = nn.Conv2d(features * 2, features, kernel_size=1, stride=1, padding=0)
            self.skip_fuse2 = nn.Conv2d(features * 2, features, kernel_size=1, stride=1, padding=0)
            self.skip_fuse1 = nn.Conv2d(features * 2, features, kernel_size=1, stride=1, padding=0)

        elif self.dpt_variant == 'hybrid_ca_shallow_concat':
            heads = _pick_num_heads(features)
            self.cross_attn4 = CrossAttention2D(features, num_heads=heads)
            self.cross_attn3 = CrossAttention2D(features, num_heads=heads)
            self.skip_fuse1 = nn.Conv2d(features * 2, features, kernel_size=1, stride=1, padding=0)

            if features % heads != 0:
                raise ValueError(f"features ({features}) must be divisible by heads ({heads})")

        head_features_1 = features
        head_features_2 = 32

        act_func = nn.Sigmoid() if output_act == 'sigmoid' else nn.Identity()

        if nclass > 2:
            # Multi-class segmentation head (e.g. semantic classes)
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass,
                          kernel_size=1, stride=1, padding=0),
            )
        else:
            # Depth head: nclass ∈ {1, 2}
            #   nclass=1 → single-channel depth (original, sigmoid output)
            #   nclass=2 → channel 0 = μ depth, channel 1 = s = log(σ) uncertainty
            #              sigmoid applied only to channel 0 in the model forward
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, nclass, kernel_size=1,
                          stride=1, padding=0),
                # Sigmoid only for single-channel depth; for nclass=2 it's applied per-channel in the model forward
                act_func if nclass == 1 else nn.Identity(),
            )

    def forward(
        self,
        out_features,
        patch_h,
        patch_w,
        prompt_depth=None,
        return_intermediate=False,
        output_size=None,
    ):
        out = []
        for i, feat in enumerate(out_features):
            if isinstance(feat, (tuple, list)):
                x = feat[0]
                if self.use_clstoken:
                    cls_token = feat[1]
                    readout = cls_token.unsqueeze(1).expand_as(x)
                    x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(
                    (x.shape[0], x.shape[-1], patch_h, patch_w))
            elif feat.ndim == 3:
                x = feat.permute(0, 2, 1).reshape(
                    (feat.shape[0], feat.shape[-1], patch_h, patch_w))
            elif feat.ndim == 4:
                expected_c = self.in_channels[i]
                if feat.shape[1] == expected_c:
                    x = feat
                elif feat.shape[-1] == expected_c:
                    x = feat.permute(0, 3, 1, 2).contiguous()
                else:
                    raise ValueError(
                        f"Feature {i} has shape {tuple(feat.shape)}, but expected "
                        f"{expected_c} channels in NCHW or NHWC layout."
                    )
            else:
                raise ValueError(f"Unsupported feature {i} shape: {tuple(feat.shape)}")

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(
            layer_4_rn, size=layer_3_rn.shape[2:], prompt_depth=prompt_depth)

        if self.dpt_variant == 'skip_concat_1x1':
            path_3_input = self.skip_fuse3(torch.cat([path_4, layer_3_rn], dim=1))
            path_3 = self.scratch.refinenet3(
                path_3_input, size=layer_2_rn.shape[2:], prompt_depth=prompt_depth)
            path_2_input = self.skip_fuse2(torch.cat([path_3, layer_2_rn], dim=1))
            path_2 = self.scratch.refinenet2(
                path_2_input, size=layer_1_rn.shape[2:], prompt_depth=prompt_depth)
            path_1_input = self.skip_fuse1(torch.cat([path_2, layer_1_rn], dim=1))
            path_1 = self.scratch.refinenet1(path_1_input, prompt_depth=prompt_depth)
        elif self.dpt_variant == 'hybrid_ca_shallow_concat':
            path_4_ca = _cross_attend_with_cap(self.cross_attn4, path_4, layer_3_rn)
            path_3 = self.scratch.refinenet3(
                path_4_ca, size=layer_2_rn.shape[2:], prompt_depth=prompt_depth)
            path_3_ca = _cross_attend_with_cap(self.cross_attn3, path_3, layer_2_rn)
            path_2 = self.scratch.refinenet2(
                path_3_ca, size=layer_1_rn.shape[2:], prompt_depth=prompt_depth)

            path_1_input = self.skip_fuse1(torch.cat([path_2, layer_1_rn], dim=1))
            path_1 = self.scratch.refinenet1(path_1_input, prompt_depth=prompt_depth)
        else:
            path_3 = self.scratch.refinenet3(
                path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_depth=prompt_depth)
            path_2 = self.scratch.refinenet2(
                path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_depth=prompt_depth)
            path_1 = self.scratch.refinenet1(
                path_2, layer_1_rn, prompt_depth=prompt_depth)

        out = self.scratch.output_conv1(path_1)
        if output_size is None:
            output_size = (int(patch_h * 14), int(patch_w * 14))
        out = F.interpolate(
            out, output_size,
            mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        if return_intermediate:
            return out, layer_4_rn
        return out
