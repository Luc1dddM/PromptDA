#!/usr/bin/env bash
set -euo pipefail

# ARKitScenes without gate network, seed 10
/home/naeem/clear/envs/env_promptda/bin/python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --ablate_gate True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_gate_network_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& /home/naeem/clear/envs/env_promptda/bin/python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --ablate_gate \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_gate_network_fair_v1_vitl_legacy_10_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_gate_network_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# ARKitScenes without gate network, seed 26
/home/naeem/clear/envs/env_promptda/bin/python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --ablate_gate True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_gate_network_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& /home/naeem/clear/envs/env_promptda/bin/python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --ablate_gate \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_gate_network_fair_v1_vitl_legacy_26_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_gate_network_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# ARKitScenes without gate network, seed 42
/home/naeem/clear/envs/env_promptda/bin/python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --ablate_gate True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_gate_network_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& /home/naeem/clear/envs/env_promptda/bin/python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --ablate_gate \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_gate_network_fair_v1_vitl_legacy_42_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_gate_network_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42
