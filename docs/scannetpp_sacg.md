# SACG on scannetpp_ours

The existing SACG trainer and comparison inference accept `--dataset scannetpp`.
ARKitScenes remains the default. Activate the existing environment first:

```bash
conda activate env_promptda
cd /home/naeem/Philipp/PromptDA
```

The default ScanNet++ root is `/media/naeem/4TB/datasets/scannetpp_ours`;
override it with `--data_root`. The loader uses:

- `splits/train.txt` and `splits/val.txt`: scene IDs (230/20 in this copy).
- `data/<scene>/iphone/rgb/frame_*.jpg`: camera RGB, converted to RGB order.
- `depth/frame_*.png`: native iPhone depth prompts (sampled files: 192×256).
- `render_depth/frame_*.png`: rendered supervision (sampled files: 1440×1920).
- `rgb_masks/frame_*.png`: nonzero pixels are valid; zero masks target and prompt.

Only frames with rendered targets are indexed. Missing paired assets raise an
error; scenes without rendered targets are skipped with an explicit warning.
This copy contains 156,456 paired training frames and 14,437 paired validation
frames. Seven training scenes and one validation scene have no rendered targets.
Train/validation scene overlap is rejected. Frame limits select a reproducible
scene-balanced, temporally stratified subset using `--data_seed` (default 26),
independent of the model seed. Initial indexing can take time on the external drive.

Both depth streams are divided by 1000 to obtain metres. Use `--depth_scale` if
your export uses different units. RGB and target are resized to 756×1008; target
and masks use nearest-neighbour interpolation. Prompts retain native resolution.
`--image_height` and `--image_width` change RGB/target resolution and must be
multiples of 14. `--ignore_rgb_masks` disables mask application. Rendered RGB is
not used as input. Training uses synchronized horizontal flips; inference does
not augment even when evaluating the training split.

## Train

```bash
python training/train_sacg.py \
  --dataset scannetpp --encoder vitl --run_name scannetpp \
  --batch_size 2 --num_workers 4 --num_iter 10000 \
  --use_wandb false --compare_after_train true --compare_save_visuals true
```

This freezes PromptDA, trains the SACG head, validates against rendered depth,
and compares the best checkpoint to the baseline after training. Official
PromptDA weights are fetched if not cached; `--promptda_ckpt /path/to/model.ckpt`
selects a local backbone checkpoint. `--resume /path/to/latest.pth` restores SACG,
optimizer and scheduler state. Keep the encoder, backbone, DPT variant and LiDAR
options consistent when resuming or running inference.

With the command above and seed 42, checkpoints are stored in
`checkpoints/scannetpp_vitl_legacy_42_sacg_/`. The trailing underscore follows the
existing trainer's directory naming. `best.pth` uses validation L1; `latest.pth`
is saved at completion. Validation also runs at the last iteration, so short
runs can produce a best checkpoint. Logs include baseline/refined metrics;
visualizations include depth, gates and LiDAR/gradient fields.

For a short pipeline check, add `--max_samples 4 --max_val_samples 2 --num_iter 2
--batch_size 1 --num_workers 0 --eval_freq 2 --log_freq 1 --visual_freq 2`.
This still uses the pretrained model and is not a meaningful training result.

## Inference and comparison

```bash
python compare_inference.py \
  --dataset scannetpp --encoder vitl --split val \
  --sacg_ckpt checkpoints/scannetpp_vitl_legacy_42_sacg_/best.pth \
  --output_dir results/scannetpp_sacg \
  --save_visuals --num_visuals 8 --save_predictions
```

Use `--baseline_ckpt` here for the same local PromptDA weights supplied during
training. Outputs include `comparison_results.json`, `comparison_results.txt`,
visualizations, and `predictions/<scene>/<frame>_{sacg,baseline}.npy`. Prediction
arrays are float depth in metres at the configured RGB size. Inference uses
paired data with rendered ground truth for evaluation. Omitting `--sacg_ckpt`
uses a random head for debugging, not a trained refinement model.

## Verification

```bash
python -m unittest discover -s tests -p test_scannetpp.py -v
```

Tests cover depth units, RGB ordering, mask handling, prompt resolution,
reproducible sampling, split leakage and missing data. A real frame also passed
a full ViT-S/SACG CPU forward/backward and inference smoke check at 56×84 using
random backbone weights and native attention (the installed xformers kernels
require CUDA). This checks connectivity and gradients, not model quality.
No full training run or pretrained GPU evaluation was performed.


## Recommended affordable research protocol

Start with **20,000 training frames and 2,000 validation frames**. This is a
compute-budget choice, not the published PromptDA frame protocol. Selection uses
approximately equal scene quotas, redistributes quotas when a scene is short,
and draws one frame from each temporal bin within a scene. All available scenes
are represented when the limit permits. Keep `--data_seed 26` fixed for every
method, ablation and training seed. Missing scenes remain disclosed in manifests.

```bash
python training/train_sacg.py \
  --dataset scannetpp --encoder vitl --run_name scannetpp_20k_fair_v1 \
  --max_samples 20000 --max_val_samples 2000 --data_seed 26 \
  --batch_size 2 --num_iter 10000 --eval_freq 2000 \
  --save_freq 5000 --visual_freq 5000 --seed 26 --learnable_lidar True \
  --compare_after_train True --compare_save_visuals True
```

Automatic comparison forwards the same data root, resolution, subset seed,
validation limit, backbone and SACG options. It uses the best validation-L1
checkpoint and writes results to the run's `compare_best` directory. Keep 756x1008
for both methods; report that resolution explicitly rather than copying numbers
from a different benchmark resolution. Both dataset loaders now accept the same
image-height/width options and preserve native LiDAR grid dimensions.
ARKitScenes 90-degree rotations are resampled back to that grid to match the
fixed landscape RGB/GT warp and permit batches with mixed sky directions.

At batch size 2, 10K steps process 20K examples. Reducing the available training
pool does not reduce the number of optimizer steps; it mainly changes coverage.
The 2K validation cap and evaluation every 2K steps substantially reduce evaluation
work. For final reporting, freeze all design choices before a larger evaluation
(e.g. 5K frames with the same selection procedure), and preferably use a separate
held-out scene split. Evaluating on more frames from validation scenes is not an
independent test set. Report all frame/scene counts and do not claim the paper's
exact protocol or zero-shot generalization without checking checkpoint provenance.

Every run saves train/validation frame manifests with hashes; comparison saves
an evaluation manifest, configuration, aggregation convention and per-frame
paired metrics. Existing manifests cannot silently change within an output
folder. Use a new run name for corrected experiments. ARKitScenes experiments
trained before the RGB/native-prompt fixes should be retrained.

Metrics use a shared finite GT mask with depth > 0.01 metres, honor supplied
validity masks, and reject non-finite predictions on valid pixels. All primary
metrics are computed per frame and averaged across frames, independently of
comparison batch size. Empty-GT frames have undefined metrics and are excluded
from the respective averages. `delta0.5` is included alongside delta1/2/3.
SSIM is an auxiliary zero-filled-image metric, not an official masked depth
benchmark metric. BoundaryAbsRel uses RGB edges, not annotated depth boundaries.

For research attribution, include a similarly trained simple refinement baseline
and component/loss ablations. Frozen PromptDA versus trained SACG alone measures
the combined effect of adaptation and architecture. Repeat final runs with
multiple training seeds while holding the subset seed fixed. Baseline and SACG
receive identical RGB/LiDAR frames and evaluation masks; comparison checks that
their frozen-backbone outputs agree. Checkpoints with missing required model
weights now fail instead of silently leaving random parameters.

The simple baseline is a parameter-matched residual CNN over RGB, frozen
PromptDA depth, resized LiDAR depth, and the LiDAR-validity mask. It has no SACG
gate, gradient-consistency input, learned LiDAR field, or DPT feature input, and
uses plain valid-pixel L1. Train seed 26 on both datasets with:

```bash
bash scripts/run_simple_refiner_baseline.sh all
```

For final three-seed results, use
`SEEDS="10 26 42" bash scripts/run_simple_refiner_baseline.sh all`.

Validation of these changes: see `tests/test_experiment_fairness.py` and
`tests/test_scannetpp.py`. Full pretrained GPU training is a separate experiment.
