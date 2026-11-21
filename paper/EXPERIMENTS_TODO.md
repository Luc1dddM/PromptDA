# Remaining experiments for publication

## Already available in experiment branch (`Lamns_v4_SACG`)
- [x] Three-seed SACG-NoBoundary on ARKitScenes and local ScanNet++ export.
- [x] Three-seed simple CNN results on both datasets.
- [x] Three-seed no-gate-network control (`G=1`) on both datasets.
- [x] Three-seed matched learned-gate/no-gate-loss control under the original boundary-weighted objective.
- [x] Paired evaluation frame manifests within each seed.

## Still needed
1. Train the **selected SACG-NoBoundary** with and without the gate network under the same no-boundary objective, rather than extrapolating the full-boundary gate ablation to the selected model.
2. Fix training and evaluation data subsets across ScanNet++ seeds; then report uncertainty across training seeds. Use paired frame-level and preferably scene-bootstrap confidence intervals.
3. Evaluate the frozen baseline, chosen adapter, and input-matched controls once on an independent scene-held-out final test set after model and loss selection.
4. Under a common objective, ablate learned LiDAR-validity field, RGB-depth gradient consistency, and frozen DPT features; compare a parameter-matched ungated network.
5. Analyze gate maps against actual depth error, distance to valid LiDAR samples, sensor sparsity/noise, and GT depth discontinuities. RGB-edge BoundaryAbsRel alone is not a geometric boundary metric.
6. Report trainable parameters, runtime, VRAM, positive-depth violations, and LiDAR-observation consistency.
7. If claiming superiority to end-to-end adaptation, train a genuine PromptDA fine-tuning baseline; current SACG and Simple CNN keep PromptDA frozen.
8. Validate source references, representative prediction figures, authorship, disclosure requirements, and target-journal template.

Do not claim state of the art, superior geometric edge preservation, calibrated confidence, or statistically established generalization using the current validation-only results.
