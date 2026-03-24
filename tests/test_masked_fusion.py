import torch

from promptda.model.masked_fusion import MaskedLocalFusion


def test_masked_local_fusion_preserves_background():
    module = MaskedLocalFusion(in_channels=4, roi_output_size=7, sampling_ratio=2)
    with torch.no_grad():
        module.projector.weight.zero_()
        module.projector.bias.fill_(1.0)

    f_global = torch.zeros(1, 4, 8, 8)
    boxes = [torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32)]

    out = module(f_global, boxes)

    # Outside top-left quadrant must remain exactly zero.
    outside = out.clone()
    outside[:, :, 0:4, 0:4] = 0
    assert torch.all(outside == 0), "Background changed outside the box mask"
