import torch
import torch.nn as nn
from promptda.model.dpt import DPTHead
from promptda.model.config import model_configs
from promptda.utils.logger import Log
import os
from pathlib import Path


class PromptDA(nn.Module):
    patch_size = 14  # patch size of the pretrained dinov2 model
    use_bn = False
    use_clstoken = False
    output_act = 'sigmoid'

    def __init__(self,
                 encoder='vitl',
                 ckpt_path=None,
                 dpt_variant='legacy'):
        super().__init__()
        model_config = model_configs[encoder]

        self.encoder = encoder
        self.model_config = model_config
        self.dpt_variant = dpt_variant
        module_path = Path(__file__) # From anywhere else: module_path = Path(inspect.getfile(PromptDA))
        package_base_dir = str(Path(*module_path.parts[:-2])) # extract path to PromptDA module, then resolve to repo base dir for dinov2 load
        self.pretrained = torch.hub.load(
            f'{package_base_dir}/torchhub/facebookresearch_dinov2_main',
            'dinov2_{:}14'.format(encoder),
            source='local',
            pretrained=True)
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(nclass=1,
                                  in_channels=dim,
                                  features=model_config['features'],
                                  out_channels=model_config['out_channels'],
                                  use_bn=self.use_bn,
                                  use_clstoken=self.use_clstoken,
                                  output_act=self.output_act,
                                  dpt_variant=self.dpt_variant)

        self._freeze_backbone_only_train_head()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        Log.info(f"Trainable: {trainable:,} / {total:,} params (DPT head only)")

        # mean and std of the pretrained dinov2 model
        self.register_buffer('_mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        encoder='vitl',
        dpt_variant='legacy',
        **hf_kwargs,
    ):
        if hf_kwargs:
            Log.warn(f"Unused kwargs in baseline from_pretrained: {list(hf_kwargs.keys())}")

        if pretrained_model_name_or_path is not None:
            Log.warn(
                "Ignoring pretrained_model_name_or_path in baseline training flow. "
                "Only DINOv2 backbone pretrained weights are loaded."
            )

        return cls(encoder=encoder, ckpt_path=None, dpt_variant=dpt_variant)


    def _freeze_backbone_only_train_head(self):
        for p in self.pretrained.parameters():
            p.requires_grad = False
        for p in self.depth_head.parameters():
            p.requires_grad = True


    def load_checkpoint(self, ckpt_path):
        if os.path.exists(ckpt_path):
            Log.info(f'Loading checkpoint from {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            if all(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items()}
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                Log.warn(f'Missing keys: {missing}')
            if unexpected:
                Log.warn(f'Unexpected keys: {unexpected}')
        else:
            Log.warn(f'Checkpoint {ckpt_path} not found')

    def trainable_parameters(self):
        return [p for p in self.depth_head.parameters() if p.requires_grad]

    def forward(self, x, prompt_depth=None):
        assert prompt_depth is not None, 'prompt_depth is required'
        prompt_depth, min_val, max_val = self.normalize(prompt_depth)
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(
            (x - self._mean) / self._std, self.model_config['layer_idxs'],
            return_class_token=True)
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        depth = self.depth_head(features, patch_h, patch_w, prompt_depth)
        depth = self.denormalize(depth, min_val, max_val)
        return depth

    @torch.no_grad()
    def predict(self,
                image: torch.Tensor,
                prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    def normalize(self,
                  prompt_depth: torch.Tensor):
        B, C, H, W = prompt_depth.shape
        min_val = torch.quantile(
            prompt_depth.reshape(B, -1), 0., dim=1, keepdim=True)[:, :, None, None]
        max_val = torch.quantile(
            prompt_depth.reshape(B, -1), 1., dim=1, keepdim=True)[:, :, None, None]
        prompt_depth = (prompt_depth - min_val) / (max_val - min_val)
        return prompt_depth, min_val, max_val

    def denormalize(self,
                    depth: torch.Tensor,
                    min_val: torch.Tensor,
                    max_val: torch.Tensor):
        return depth * (max_val - min_val) + min_val