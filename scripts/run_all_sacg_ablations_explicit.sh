#!/usr/bin/env bash
set -euo pipefail

# arkitscenes, fixed_lidar, seed 10
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar False \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_fixed_lidar_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_fixed_lidar_fair_v1_vitl_legacy_10_sacg_fixed-lidar/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_fixed_lidar_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# arkitscenes, no_lidar, seed 10
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --ablate_lidar True \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_lidar_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --ablate_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_lidar_fair_v1_vitl_legacy_10_sacg_no-lidar/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_lidar_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# arkitscenes, no_boundary, seed 10
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --use_boundary_loss False \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_boundary_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_boundary_fair_v1_vitl_legacy_10_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_boundary_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# arkitscenes, no_gate, seed 10
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_gate_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_gate_fair_v1_vitl_legacy_10_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_gate_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# arkitscenes, fixed_lidar, seed 26
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar False \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_fixed_lidar_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_fixed_lidar_fair_v1_vitl_legacy_26_sacg_fixed-lidar/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_fixed_lidar_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# arkitscenes, no_lidar, seed 26
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --ablate_lidar True \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_lidar_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --ablate_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_lidar_fair_v1_vitl_legacy_26_sacg_no-lidar/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_lidar_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# arkitscenes, no_boundary, seed 26
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --use_boundary_loss False \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_boundary_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_boundary_fair_v1_vitl_legacy_26_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_boundary_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# arkitscenes, no_gate, seed 26
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_gate_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_gate_fair_v1_vitl_legacy_26_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_gate_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# arkitscenes, fixed_lidar, seed 42
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar False \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_fixed_lidar_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_fixed_lidar_fair_v1_vitl_legacy_42_sacg_fixed-lidar/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_fixed_lidar_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# arkitscenes, no_lidar, seed 42
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --ablate_lidar True \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_lidar_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --ablate_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_lidar_fair_v1_vitl_legacy_42_sacg_no-lidar/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_lidar_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# arkitscenes, no_boundary, seed 42
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --use_boundary_loss False \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_boundary_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_boundary_fair_v1_vitl_legacy_42_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_boundary_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# arkitscenes, no_gate, seed 42
python training/train_sacg.py \
  --dataset arkitscenes \
  --learnable_lidar True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_arkitscenes_ablation_no_gate_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_arkitscenes_ablation_no_gate_fair_v1_vitl_legacy_42_sacg_learnable/best.pth \
  --save_visuals \
  --output_dir results/sacg_arkitscenes_ablation_no_gate_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# scannetpp, fixed_lidar, seed 10
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar False \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_fixed_lidar_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset scannetpp \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_fixed_lidar_fair_v1_vitl_legacy_10_sacg_fixed-lidar/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_fixed_lidar_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# scannetpp, no_lidar, seed 10
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --ablate_lidar True \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_lidar_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --ablate_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_lidar_fair_v1_vitl_legacy_10_sacg_no-lidar/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_lidar_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# scannetpp, no_boundary, seed 10
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --use_boundary_loss False \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_boundary_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_boundary_fair_v1_vitl_legacy_10_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_boundary_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# scannetpp, no_gate, seed 10
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_gate_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_gate_fair_v1_vitl_legacy_10_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_gate_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# scannetpp, fixed_lidar, seed 26
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar False \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_fixed_lidar_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset scannetpp \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_fixed_lidar_fair_v1_vitl_legacy_26_sacg_fixed-lidar/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_fixed_lidar_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# scannetpp, no_lidar, seed 26
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --ablate_lidar True \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_lidar_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --ablate_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_lidar_fair_v1_vitl_legacy_26_sacg_no-lidar/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_lidar_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# scannetpp, no_boundary, seed 26
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --use_boundary_loss False \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_boundary_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_boundary_fair_v1_vitl_legacy_26_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_boundary_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# scannetpp, no_gate, seed 26
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_gate_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_gate_fair_v1_vitl_legacy_26_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_gate_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# scannetpp, fixed_lidar, seed 42
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar False \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_fixed_lidar_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset scannetpp \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_fixed_lidar_fair_v1_vitl_legacy_42_sacg_fixed-lidar/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_fixed_lidar_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# scannetpp, no_lidar, seed 42
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --ablate_lidar True \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_lidar_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --ablate_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_lidar_fair_v1_vitl_legacy_42_sacg_no-lidar/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_lidar_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# scannetpp, no_boundary, seed 42
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --use_boundary_loss False \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_boundary_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_boundary_fair_v1_vitl_legacy_42_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_boundary_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# scannetpp, no_gate, seed 42
python training/train_sacg.py \
  --dataset scannetpp \
  --learnable_lidar True \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name sacg_scannetpp_ablation_no_gate_fair_v1 \
  --max_samples 20000 \
  --max_val_samples 2000 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 2000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset scannetpp \
  --learnable_lidar \
  --sacg_ckpt checkpoints/sacg_scannetpp_ablation_no_gate_fair_v1_vitl_legacy_42_sacg_learnable/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/sacg_scannetpp_ablation_no_gate_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42
