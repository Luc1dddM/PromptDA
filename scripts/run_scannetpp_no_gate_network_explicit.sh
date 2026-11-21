#!/usr/bin/env bash
set -euo pipefail

# ScanNet++ without gate network, seed 10
/home/naeem/clear/envs/env_promptda/bin/python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --ablate_gate True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_gate_network_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& /home/naeem/clear/envs/env_promptda/bin/python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --ablate_gate \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_gate_network_fair_v1_vitl_legacy_10_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_gate_network_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# ScanNet++ without gate network, seed 26
/home/naeem/clear/envs/env_promptda/bin/python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --ablate_gate True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_gate_network_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& /home/naeem/clear/envs/env_promptda/bin/python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --ablate_gate \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_gate_network_fair_v1_vitl_legacy_26_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_gate_network_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# ScanNet++ without gate network, seed 42
/home/naeem/clear/envs/env_promptda/bin/python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --ablate_gate True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_gate_network_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& /home/naeem/clear/envs/env_promptda/bin/python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --ablate_gate \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_gate_network_fair_v1_vitl_legacy_42_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_gate_network_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42
