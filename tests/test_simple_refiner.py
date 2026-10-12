import torch

from promptda.sacg import SACGModule, SimpleRefinementModule


def test_simple_refiner_starts_from_frozen_promptda_prediction():
    model = SimpleRefinementModule(hidden_channels=8).eval()
    rgb = torch.rand(2, 3, 28, 42)
    coarse = torch.rand(2, 1, 28, 42)
    sparse = torch.rand(2, 1, 7, 14)
    sparse[:, :, :2, :3] = 0

    output = model(rgb=rgb, sparse=sparse, d_coarse=coarse)

    torch.testing.assert_close(output["refined_depth"], coarse)
    torch.testing.assert_close(output["delta_depth"], torch.zeros_like(coarse))
    assert output["f_lidar"].shape == coarse.shape
    assert set(torch.unique(output["f_lidar"]).tolist()).issubset({0.0, 1.0})


def test_simple_refiner_is_parameter_matched_to_vitl_sacg_head():
    simple = SimpleRefinementModule(hidden_channels=56)
    sacg = SACGModule(dpt_channels=128, learnable_lidar=True)
    simple_params = sum(parameter.numel() for parameter in simple.parameters())
    sacg_params = sum(parameter.numel() for parameter in sacg.parameters())

    assert 0.8 <= simple_params / sacg_params <= 1.2
