model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'layer_idxs': [2, 5, 8, 11]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'layer_idxs': [2, 5, 8, 11]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'layer_idxs': [4, 11, 17, 23]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'layer_idxs': [9, 19, 29, 39]},
    'swint': {
        'timm_name': 'swin_tiny_patch4_window7_224',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
        'backbone_kwargs': {'img_size': (756, 1008)},
    },
    'swins': {
        'timm_name': 'swin_small_patch4_window7_224',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
        'backbone_kwargs': {'img_size': (756, 1008)},
    },
    'convnextt': {
        'timm_name': 'convnext_tiny.fb_in22k',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
    },
    'convnexts': {
        'timm_name': 'convnext_small.fb_in22k',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
    },
}
