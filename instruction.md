# Copilot Implementation Guide: Masked Local Fusion (MLF) for Prompt-Depth-Anything

This document is a natural-language instruction contract for Copilot/Cursor. Do not generate code until you have read every section. Follow the sections in order.

---

## 0. What You Are Building

The goal is to sharpen depth predictions inside object regions by fusing spatially-precise local features back into the global backbone feature map. A binary spatial mask derived from bounding boxes ensures that this fusion only affects pixels inside detected objects — the background is never modified.

The enhancement follows this logic: take the global feature map from the backbone, extract object-region features from it using RoI Align, project those features to match the global channel dimension, zero out everything outside the bounding boxes using a mask, then add the result back to the global feature map as a residual.

There are three rules that must never be violated. First, never run the backbone a second time on cropped images — all local features must come from the already-computed global feature map. Second, the spatial mask must strictly zero out any contribution outside the bounding boxes. Third, the fusion must always be additive (residual), never replacing the original features.

---

## 1. Stage 0 — Object Detection (Prerequisite)

Before any depth feature fusion can happen, the pipeline needs bounding boxes. This is an entirely separate stage that must be added before the depth model runs.

### Recommended approach: Offline detection with YOLOv8

Use a pretrained YOLOv8 model to pre-compute bounding boxes for every image in the dataset and save them alongside the images (for example, as `.json` files in the same folder structure). During training and inference, the dataloader reads these saved boxes instead of running the detector live. This keeps VRAM usage low and training speed high.

### What Copilot should implement for this stage

- Add a standalone detection script (not part of the model) that accepts a dataset directory, runs YOLOv8 on every image, and saves the bounding boxes in feature-map coordinate space to a sidecar file per image.
- The boxes must be saved in xyxy format and already scaled to the feature map resolution (not the original image resolution), because they will be passed directly to RoI Align with `spatial_scale=1.0`.
- The scaling factor is: feature map height divided by image height (and same for width). For a ViT-B/14 backbone processing a 518×518 image, the feature map is typically 37×37, so the scale factor is approximately 37/518.
- In the dataloader, load the sidecar box file that corresponds to each image. If no sidecar file exists for an image, pass an empty list of boxes — the MLF module must handle this gracefully and simply return the global features unchanged.

### Why not online detection

Running a detector inside the training loop doubles the GPU workload and makes batching harder. Offline detection is the standard approach in two-stage pipelines and is strongly recommended here.

---

## 2. Stage 1 — Expose Intermediate Features from the Backbone

The MLF module needs access to an intermediate feature map from the DINOv2/ViT backbone, not just the final depth output.

### What Copilot should do

- In `depth_anything_v2/dpt.py`, find the forward method of the DPT model or head where the backbone produces its layered outputs.
- Add an optional flag to the forward method (for example `return_intermediate`) that, when enabled, also returns one of the intermediate layer outputs alongside the final depth prediction.
- The recommended intermediate layer to expose is the deepest one (layer 4 in a 4-stage DPT), as it contains the most semantic information. This will serve as `F_global` for the MLF module.
- Do not change any of the existing DPT fusion logic. This is a non-destructive addition only.

---

## 3. Stage 2 — The MaskedLocalFusion Module

Create a new file at `models/masked_fusion.py` and define a PyTorch module called `MaskedLocalFusion` inside it.

### What the module must do, step by step

**Step A — Feature Extraction via RoI Align**
Use `torchvision.ops.roi_align` to extract fixed-size feature patches from `F_global` at the locations specified by the bounding boxes. The boxes are already in feature-map coordinates, so `spatial_scale` should be `1.0`. The sampling ratio must be consistent with the backbone's patch size — for ViT-B/14 (patch size 14), a sampling ratio of 2 is appropriate. The output will be a collection of small fixed-size patches, one per detected object across the batch.

**Step B — Spatial Reconstruction**
Create a zero-initialized tensor with the same shape as `F_global`. Scatter each extracted patch back into the spatial location corresponding to its bounding box by resizing the patch to fill the box region. Then apply a 1×1 convolution (the Projector) to the entire reconstructed tensor to align the channel dimension and learn a fusion weight.

**Step C — Masked Fusion**
Build a binary spatial mask with the same height and width as `F_global`. Every pixel that falls inside any bounding box gets the value 1.0; everything else stays 0.0. Multiply the projected local features element-wise by this mask. Then add the masked result to `F_global` as a residual connection. The output is `F_enhanced`, which has the same shape as `F_global`.

### Module configuration

The module should accept three parameters at initialization: the number of input channels (matching the backbone's intermediate feature channels), the RoI Align output size (7 is a sensible default), and the sampling ratio.

### Edge cases Copilot must handle

- If a batch image has no detected objects (empty box list), the module must return the original `F_global` for that image unchanged, with no error.
- Bounding box coordinates that extend beyond the feature map boundaries must be clamped before scattering or masking.

---

## 4. Stage 3 — Integration into the Prompt Encoder

Edit `prompt_da/models/prompt_encoder.py` to wire the MLF module into the existing pipeline.

### What Copilot should do

- Import `MaskedLocalFusion` from `models/masked_fusion`.
- Instantiate it inside `__init__` using the channel dimension of the backbone's intermediate feature map.
- In the `forward` method, call the backbone with `return_intermediate=True` to get both the depth prediction and `F_global`.
- If bounding boxes are provided, pass `F_global` through `self.mlf` to get `F_enhanced`. If no boxes are provided, use `F_global` as-is.
- Use the resulting feature map (enhanced or not) for all subsequent steps: point-prompt embedding and the DPT decoder input.
- The `forward` method signature should accept an optional `boxes` argument (list of tensors, one per image) that defaults to `None`.

### Injection point

The MLF call must happen between backbone feature extraction and the DPT decoder stages. It must not be placed inside the decoder or before the backbone.

---

## 5. Training Strategy

### Phase 1 — Freeze backbone, train MLF projector only
In the first training phase, freeze all backbone parameters. Only the MLF projector (the 1×1 convolution) should have gradients enabled. Train for a small number of epochs to let the projector learn a stable fusion weight before the backbone adapts.

### Phase 2 — Unfreeze and fine-tune jointly
Unfreeze the backbone and train the full model jointly with a lower learning rate for the backbone than for the MLF module and decoder.

### Loss function
Do not change the existing Scale-and-Shift Invariant depth loss used by Prompt-DA. MLF is a feature-level change and does not require a new loss term.

---

## 6. VRAM Constraints — Copilot Must Follow All of These

- Never crop the original image and re-run the backbone. Extract features from the existing `F_global` only.
- The zero-initialized reconstruction canvas must be created once per forward call, not inside any inner loop.
- Use bilinear interpolation when resizing patches back to box size. Do not use bicubic.
- The spatial mask must be a float tensor, not a boolean tensor, to avoid implicit type promotion during multiplication.
- Confirm that `torch.no_grad()` wraps the backbone during inference, including the `return_intermediate` path.

---

## 7. Validation

After implementation, write a single unit test in `tests/test_masked_fusion.py` that does the following without any code dependencies on the rest of the project. Create a dummy zero-valued global feature map and a single bounding box covering only the top-left quadrant. Run `MaskedLocalFusion` on it. Assert that every value outside the bounding box in the output remains exactly zero. This confirms the mask is working and no feature is leaking into the background.

---

## 8. File Summary

| File | Action | Purpose |
|---|---|---|
| `scripts/precompute_boxes.py` | Create | Offline YOLOv8 detection; saves boxes per image as sidecar files |
| `models/masked_fusion.py` | Create | MaskedLocalFusion module |
| `depth_anything_v2/dpt.py` | Edit | Expose intermediate feature map via optional flag |
| `prompt_da/models/prompt_encoder.py` | Edit | Instantiate and call MLF between backbone and decoder |
| `tests/test_masked_fusion.py` | Create | Background-leaking sanity check |

---

*Masked Local Fusion Implementation Contract v1.1 — instruction-only*