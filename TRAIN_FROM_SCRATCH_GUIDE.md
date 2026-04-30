# Train From Scratch (No MLF) Guide

This guide explains how to train PromptDA **from scratch** without using MLF.

## 1) Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure your ARKitScenes data is available at:

```text
data/ARKitScenes/data/upsampling
```

If your dataset is in another location, pass it with `--data_root`.

## 2) Recommended training command

Run this from the project root:

```bash
python training/train.py \
  --from_scratch true \
  --use_mlf false \
  --run_name scratch_no_mlf \
  --encoder vitl \
  --data_root data\ARKitScenes\data\upsampling \
  --epochs 20 \
  --batch_size 4 \
  --lr_backbone 1e-5 \
  --lr_head 1e-4 \
  --num_workers 4 \
  --seed 42
```

## 3) What each important flag means

- `--from_scratch true`: train model weights from scratch path (no pretrained PromptDA checkpoint loading).
- `--use_mlf false`: disable MLF module.
- `--encoder vitl`: choose backbone size (`vits`, `vitb`, or `vitl`).
- `--lr_backbone`: learning rate for DINOv2 backbone.
- `--lr_head`: learning rate for DPT depth head.
- `--run_name`: run identifier used in checkpoint folder naming.

## 4) Output files

Checkpoints and logs are written under:

```text
./checkpoints/{run_name}_{encoder}_{seed}/
```

Typical files:
- `best.pth`
- `latest.pth`
- `metrics_history.json`
- `training_curves.png`

## 5) Resume training

Resume from a checkpoint:

```bash
python training/train.py \
  --from_scratch true \
  --use_mlf false \
  --run_name scratch_no_mlf \
  --encoder vitl \
  --data_root data/ARKitScenes/data/upsampling \
  --epochs 20 \
  --batch_size 4 \
  --lr_backbone 1e-5 \
  --lr_head 1e-4 \
  --num_workers 4 \
  --seed 42 \
  --resume ./checkpoints/scratch_no_mlf_vitl_42/latest.pth
```

## 6) Optional: W&B logging

Enable Weights & Biases:

```bash
python training/train.py \
  --from_scratch true \
  --use_mlf false \
  --run_name scratch_no_mlf \
  --encoder vitl \
  --data_root data/ARKitScenes/data/upsampling \
  --epochs 20 \
  --batch_size 4 \
  --lr_backbone 1e-5 \
  --lr_head 1e-4 \
  --num_workers 4 \
  --seed 42 \
  --use_wandb true \
  --wandb_mode online
```

## 7) Quick troubleshooting

- If you hit OOM: lower `--batch_size` (for example to `2` or `1`).
- If training is unstable: reduce `--lr_backbone` (for example `5e-6`).
- If data path error appears: verify `--data_root` points to your ARKitScenes upsampling folder.
