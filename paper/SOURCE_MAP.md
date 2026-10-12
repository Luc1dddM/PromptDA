# Research-paper source map

Model and architecture: `promptda/sacg.py` (`SACGModule`, `PromptDASACG`, `SimpleRefinementModule`).

Loss and optimization: `promptda/sacg.py` (`SACGLoss`) and `training/train_sacg.py` (AdamW over `model.sacg.parameters()`, frozen PromptDA).

Metrics: `training/metrics.py`; dataset manifests: `dataset/experiment.py`.

ARKitScenes selected no-boundary, seed 10: `results/sacg_arkitscenes_ablation_no_boundary_fair_v1_seed_10/comparison_results.json`.

ARKitScenes full SACG seed 10: `results/arkit_sacg_all_fair_v1_seed_10_learnable/comparison_results.json`.

Simple residual CNN seed 10: `results/simple_refiner_arkitscenes_fair_v1_seed_10/comparison_results.json`.

No LiDAR gate field: `results/sacg_arkitscenes_ablation_no_lidar_fair_v1_seed_10/comparison_results.json`.

Fixed LiDAR gate field: `results/sacg_arkitscenes_ablation_fixed_lidar_fair_v1_seed_10/comparison_results.json`.

Full-objective other seeds: `results/arkit_sacg_all_fair_v1_seed_{26,42}_learnable/comparison_results.json`.

Full-objective ScanNet++: `results/scannetpp_20k_fair_v1_seed_{10,26,42}_learnable/comparison_results.json`.

No new model training or independent evaluation was performed while preparing the manuscript. Full SACG and selected no-boundary SACG use different training objectives and must not be pooled.