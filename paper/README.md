# SACG manuscript — updated gate-network ablation (November 2026)

This is a working, venue-neutral LaTeX manuscript (IEEE Access is a tentative target, not the current official document class).

Compile in `paper/` with:

```bash
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

**Current central claim:** Under matched residual inputs and boundary-weighted training objectives, a learned spatial gate improves AbsRel over the same residual with `G=1` in all three ARKitScenes seed runs and all three paired ScanNet++ seed/subset runs. The mean relative AbsRel improvements against the ungated counterpart are 0.70% and 1.03%, respectively. Importantly, both gated and ungated variants in this *architectural ablation* have **gate regularization disabled** and retain **boundary-weighted supervision**.

**Selected accuracy-focused configuration:** SACG-NoB removes the boundary loss (but retains gate regularization). It improves three-seed mean AbsRel over frozen PromptDA by 1.63% on ARKitScenes and 0.64% on the local ScanNet++ export. The constant-gate result above **does not isolate** gate contribution specifically under this selected loss and must not be used to imply it does.

**CNN comparisons:** The simple CNN has slightly better mean metrics than SACG-NoB on ARKitScenes across three seeds. ScanNet++ results are mixed and metric-dependent. The CNN uses a different feature interface, so the gate's causal effect should be attributed to the matched same-residual ablation, not the CNN comparison.

The writing branch is `Lamns_V4_SACG` (uppercase V4). The up-to-date results and code are on `Lamns_v4_SACG` (lowercase v4), head `bc9b750b8eba8bdddf3bdd7c079029167961264c` when this revision was prepared. The paper cites stored evaluation JSON from the experiment branch without copying experiment files into the manuscript branch or rerunning the checkpoints.

All results are validation results, not independent held-out test scores; ScanNet++ uses different 2,000-frame subsets for seeds 10/26/42. The manuscript still requires independent tests, mechanism-based spatial analysis, resource/latency measurement, and migration to an actual journal template before submission. See `SOURCE_MAP.md` and `EXPERIMENTS_TODO.md` for details.
