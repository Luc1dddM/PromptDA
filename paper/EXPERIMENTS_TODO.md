# Outstanding paper experiments
1. Train the selected no-boundary SACG with seeds 10, 26, 42 and a fixed train/eval data subset seed. Evaluate on identical frames and report mean +/- standard deviation.
2. Retrain the simple CNN on exactly matched frames, seeds and budget; calculate paired frame-level and preferably scene-bootstrap confidence intervals.
3. Run selected no-boundary SACG on ScanNet++ using fixed subsets, rather than attributing existing full-objective results to it.
4. Under the selected objective: ablate gate by replacing it with 1; ablate gate loss; learned LiDAR validity field; gradient consistency; DPT features.
5. Add an independent scene-held-out test set after all decisions are frozen; avoid treating model-selection validation metrics as final generalization evidence.
6. Quantify model complexity, runtime, memory, observed LiDAR consistency, positivity violations and geometry-based depth edge quality.
7. If required for a publication claim, train a genuine PromptDA fine-tuning baseline with explicitly identified trainable weights. Neither the current SimpleRefiner nor SACG fine-tunes PromptDA.
8. Finalize actual prediction figures, authorship, target venue style and external reference checking.