#!/usr/bin/env bash
set -euo pipefail

# Train the plain residual-CNN baseline under the same protocol as full SACG.
# Usage:
#   bash scripts/run_simple_refiner_baseline.sh arkitscenes
#   bash scripts/run_simple_refiner_baseline.sh scannetpp
#   bash scripts/run_simple_refiner_baseline.sh all
# Three final seeds:
#   SEEDS="10 26 42" bash scripts/run_simple_refiner_baseline.sh all

dataset="${1:-all}"
seeds="${SEEDS:-26}"

case "$dataset" in
  arkitscenes|scannetpp|all) ;;
  *)
    echo "Unknown dataset: $dataset (use arkitscenes, scannetpp, or all)" >&2
    exit 2
    ;;
esac

datasets=(arkitscenes scannetpp)
if [[ "$dataset" != "all" ]]; then
  datasets=("$dataset")
fi

for current_dataset in "${datasets[@]}"; do
  dataset_args=(--dataset "$current_dataset")
  eval_freq=1000
  if [[ "$current_dataset" == "scannetpp" ]]; then
    dataset_args+=(--max_samples 20000 --max_val_samples 2000)
    eval_freq=2000
  fi

  for seed in $seeds; do
    python training/train_sacg.py \
      "${dataset_args[@]}" \
      --refiner simple \
      --simple_hidden_channels 56 \
      --encoder vitl \
      --run_name "simple_refiner_${current_dataset}_fair_v1" \
      --batch_size 2 \
      --num_iter 10000 \
      --eval_freq "$eval_freq" \
      --save_freq 5000 \
      --visual_freq 5000 \
      --seed "$seed" \
      --data_seed 26 \
      --use_boundary_loss False \
      --gate_weight 0.0 \
      --compare_after_train True \
      --compare_save_visuals False
  done
done
