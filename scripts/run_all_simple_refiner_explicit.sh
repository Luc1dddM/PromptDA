#!/usr/bin/env bash
set -euo pipefail

# arkitscenes, simple_refiner, seed 10
python training/train_sacg.py \
  --dataset arkitscenes \
  --refiner simple \
  --simple_hidden_channels 56 \
  --use_boundary_loss False \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name simple_refiner_arkitscenes_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 10 \
  --data_seed 10 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --refiner simple \
  --simple_hidden_channels 56 \
  --sacg_ckpt checkpoints/simple_refiner_arkitscenes_fair_v1_vitl_legacy_10_simple/best.pth \
  --save_visuals \
  --output_dir results/simple_refiner_arkitscenes_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# arkitscenes, simple_refiner, seed 26
python training/train_sacg.py \
  --dataset arkitscenes \
  --refiner simple \
  --simple_hidden_channels 56 \
  --use_boundary_loss False \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name simple_refiner_arkitscenes_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 26 \
  --data_seed 26 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --refiner simple \
  --simple_hidden_channels 56 \
  --sacg_ckpt checkpoints/simple_refiner_arkitscenes_fair_v1_vitl_legacy_26_simple/best.pth \
  --save_visuals \
  --output_dir results/simple_refiner_arkitscenes_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# arkitscenes, simple_refiner, seed 42
python training/train_sacg.py \
  --dataset arkitscenes \
  --refiner simple \
  --simple_hidden_channels 56 \
  --use_boundary_loss False \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name simple_refiner_arkitscenes_fair_v1 \
  --batch_size 2 \
  --num_iter 10000 \
  --eval_freq 1000 \
  --save_freq 5000 \
  --visual_freq 5000 \
  --seed 42 \
  --data_seed 42 \
&& python compare_inference.py \
  --dataset arkitscenes \
  --refiner simple \
  --simple_hidden_channels 56 \
  --sacg_ckpt checkpoints/simple_refiner_arkitscenes_fair_v1_vitl_legacy_42_simple/best.pth \
  --save_visuals \
  --output_dir results/simple_refiner_arkitscenes_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42

# scannetpp, simple_refiner, seed 10
python training/train_sacg.py \
  --dataset scannetpp \
  --refiner simple \
  --simple_hidden_channels 56 \
  --use_boundary_loss False \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name simple_refiner_scannetpp_20k_fair_v1 \
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
  --refiner simple \
  --simple_hidden_channels 56 \
  --sacg_ckpt checkpoints/simple_refiner_scannetpp_20k_fair_v1_vitl_legacy_10_simple/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/simple_refiner_scannetpp_20k_fair_v1_seed_10 \
  --encoder vitl \
  --seed 10 \
  --data_seed 10

# scannetpp, simple_refiner, seed 26
python training/train_sacg.py \
  --dataset scannetpp \
  --refiner simple \
  --simple_hidden_channels 56 \
  --use_boundary_loss False \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name simple_refiner_scannetpp_20k_fair_v1 \
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
  --refiner simple \
  --simple_hidden_channels 56 \
  --sacg_ckpt checkpoints/simple_refiner_scannetpp_20k_fair_v1_vitl_legacy_26_simple/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/simple_refiner_scannetpp_20k_fair_v1_seed_26 \
  --encoder vitl \
  --seed 26 \
  --data_seed 26

# scannetpp, simple_refiner, seed 42
python training/train_sacg.py \
  --dataset scannetpp \
  --refiner simple \
  --simple_hidden_channels 56 \
  --use_boundary_loss False \
  --gate_weight 0.0 \
  --encoder vitl \
  --run_name simple_refiner_scannetpp_20k_fair_v1 \
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
  --refiner simple \
  --simple_hidden_channels 56 \
  --sacg_ckpt checkpoints/simple_refiner_scannetpp_20k_fair_v1_vitl_legacy_42_simple/best.pth \
  --max_samples 2000 \
  --save_visuals \
  --output_dir results/simple_refiner_scannetpp_20k_fair_v1_seed_42 \
  --encoder vitl \
  --seed 42 \
  --data_seed 42
