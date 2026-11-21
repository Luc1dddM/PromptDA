# SACG manuscript: evidence and source map (updated November 2026)

Source-of-truth experiment branch: `Lamns_v4_SACG` (lowercase v4), commit `bc9b750b8eba8bdddf3bdd7c079029167961264c`. Writing branch: `Lamns_V4_SACG` (uppercase V4). These are different Git refs.

Implementation: `promptda/sacg.py` (`SACGModule`, `SimpleRefinementModule`, `SACGLoss`); training: `training/train_sacg.py`; evaluation: `compare_inference.py`; metrics: `training/metrics.py`.

Every line below denotes a seed-dependent path with `{seed}` replaced by 10, 26 or 42, unless indicated otherwise. All evaluation files cited here live on the experiment branch, not the writing branch.

- ARKitScenes SACG-NoBoundary: `results/sacg_arkitscenes_ablation_no_boundary_fair_v1_seed_{seed}/comparison_results.json`.
- ScanNet++ SACG-NoBoundary: `results/sacg_scannetpp_ablation_no_boundary_fair_v1_seed_{seed}/comparison_results.json`.
- ARKitScenes no gate *network* (`G=1`): `results/sacg_arkitscenes_ablation_no_gate_network_fair_v1_seed_{seed}/comparison_results.json`.
- ScanNet++ no gate *network* (`G=1`): `results/sacg_scannetpp_ablation_no_gate_network_fair_v1_seed_{seed}/comparison_results.json`.
- ARKitScenes no gate *loss*, with learned gate still present: `results/sacg_arkitscenes_ablation_no_gate_fair_v1_seed_{seed}/comparison_results.json`.
- ScanNet++ no gate *loss*, with learned gate still present: `results/sacg_scannetpp_ablation_no_gate_fair_v1_seed_{seed}/comparison_results.json`.
- Simple CNN ARKitScenes: `results/simple_refiner_arkitscenes_fair_v1_seed_{seed}/comparison_results.json`.
- Simple CNN ScanNet++: `results/simple_refiner_scannetpp_20k_fair_v1_seed_{seed}/comparison_results.json`.
- Full-objective SACG ARKitScenes: `results/arkit_sacg_all_fair_v1_seed_{seed}_learnable/comparison_results.json`.
- Full-objective SACG ScanNet++: `results/scannetpp_20k_fair_v1_seed_{seed}_learnable/comparison_results.json`.
- Additional LiDAR gate-field controls: corresponding `fixed_lidar` and `no_lidar` ablation folders.

The matched gate-network attribution compares no-gate-loss versus no-gate-network under **the original boundary-weighted training objective**. It does not directly attribute SACG-NoBoundary improvements to the gate.

ARKitScenes 5,599-frame evaluation manifest is identical for all seeds. ScanNet++ uses 2,000 frames per seed; `data_seed=seed`, so subsets differ among seed runs. All metrics are per-image means aggregated across frames and then averaged across three runs. We have not independently rerun model checkpoints for this revision.
