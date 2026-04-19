import sys
import torch

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)

from data.ARKitScenes.depth_upsampling.losses.l1_loss import l1_loss
from data.ARKitScenes.depth_upsampling.losses.rmse import rmse_loss


@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    valid_mask = (gt > 0)

    inputs = {
        arkit_dataset_keys.HIGH_RES_DEPTH_IMG: gt,
        arkit_dataset_keys.VALID_MASK_IMG: valid_mask,
    }
    outputs = {
        arkit_dataset_keys.PREDICTION_DEPTH_IMG: pred,
    }

    l1_value = l1_loss(outputs, inputs).item()
    rmse_value = rmse_loss(outputs, inputs).item()

    return {
        "L1": l1_value,
        "RMSE": rmse_value,
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}
