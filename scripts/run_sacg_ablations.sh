#!/usr/bin/env bash
set -euo pipefail

# Core SACG ablations for the journal experiments.
#
# Usage:
#   bash scripts/run_sacg_ablations.sh arkitscenes fixed_lidar
#   bash scripts/run_sacg_ablations.sh scannetpp no_boundary
#   bash scripts/run_sacg_ablations.sh all all
#
# The default is seed 26. Repeat the final ablations with three seeds using:
#   SEEDS="10 26 42" bash scripts/run_sacg_ablations.sh all all

dataset="${1:-all}"
ablation="${2:-all}"

encoder="${ENCODER:-vitl}"
batch_size="${BATCH_SIZE:-2}"
num_iter="${NUM_ITER:-10000}"
data_seed="${DATA_SEED:-26}"
seeds="${SEEDS:-26}"
num_workers="${NUM_WORKERS:-4}"
wandb_mode="${WANDB_MODE:-online}"

# Match the completed ScanNet++ fair runs. Set either variable to an empty
# string to remove that limit, but do so only if the full SACG reference was
# trained and evaluated with the same unrestricted protocol.
scannet_train_samples="${SCANNET_TRAIN_SAMPLES:-20000}"
scannet_val_samples="${SCANNET_VAL_SAMPLES:-2000}"

case "$dataset" in
  arkitscenes|scannetpp|all) ;;
  *)
    echo "Unknown dataset: $dataset (use arkitscenes, scannetpp, or all)" >&2
    exit 2
    ;;
esac

case "$ablation" in
  fixed_lidar|no_lidar|no_boundary|no_gate|all) ;;
  *)
    echo "Unknown ablation: $ablation (use fixed_lidar, no_lidar, no_boundary, no_gate, or all)" >&2
    exit 2
    ;;
esac

run_one() {
  local current_dataset="$1"
  local current_ablation="$2"
  local seed="$3"
  local run_name="sacg_${current_dataset}_ablation_${current_ablation}_fair_v1"
  local eval_freq=1000
  local -a dataset_args=(--dataset "$current_dataset")
  local -a ablation_args=(--learnable_lidar True)

  if [[ "$current_dataset" == "scannetpp" ]]; then
    eval_freq=2000
    if [[ -n "$scannet_train_samples" ]]; then
      dataset_args+=(--max_samples "$scannet_train_samples")
    fi
    if [[ -n "$scannet_val_samples" ]]; then
      dataset_args+=(--max_val_samples "$scannet_val_samples")
    fi
  fi

  case "$current_ablation" in
    fixed_lidar)
      ablation_args=(--learnable_lidar False)
      ;;
    no_lidar)
      ablation_args+=(--ablate_lidar True)
      ;;
    no_boundary)
      ablation_args+=(--use_boundary_loss False)
      ;;
    no_gate)
      ablation_args+=(--gate_weight 0.0)
      ;;
  esac

  echo "Starting dataset=$current_dataset ablation=$current_ablation seed=$seed"
  python training/train_sacg.py \
    "${dataset_args[@]}" \
    --encoder "$encoder" \
    --run_name "$run_name" \
    --batch_size "$batch_size" \
    --num_iter "$num_iter" \
    --eval_freq "$eval_freq" \
    --save_freq 5000 \
    --visual_freq 5000 \
    --seed "$seed" \
    --data_seed "$data_seed" \
    --num_workers "$num_workers" \
    --wandb_mode "$wandb_mode" \
    --compare_after_train True \
    --compare_save_visuals False \
    "${ablation_args[@]}"
}

datasets=(arkitscenes scannetpp)
ablations=(fixed_lidar no_lidar no_boundary no_gate)

if [[ "$dataset" != "all" ]]; then
  datasets=("$dataset")
fi
if [[ "$ablation" != "all" ]]; then
  ablations=("$ablation")
fi

for current_dataset in "${datasets[@]}"; do
  for current_ablation in "${ablations[@]}"; do
    for seed in $seeds; do
      run_one "$current_dataset" "$current_ablation" "$seed"
    done
  done
done
