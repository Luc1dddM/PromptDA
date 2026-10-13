# SACG journal manuscript (IEEE Access target)

This repository contains a revised and expanded LaTeX manuscript. IEEE Access is a tentative journal target. The current main.tex still uses a generic two-column article class: it is NOT the official IEEE Access submission template.

To compile from the paper directory:

    latexmk -pdf -interaction=nonstopmode -file-line-error main.tex

A TeX distribution with TikZ, natbib, and the standard math packages is needed. Keep all sections/ files next to main.tex.

The main working SACG method retains the learned LiDAR-validity field and delayed gate regularization but omits the boundary-weighted loss. The best supported comparison is one seed on 5,599 ARKitScenes validation frames. The result is metric-specific, not established across seeds or held-out scenes.

Existing three-seed ARKitScenes and ScanNet++ numbers correspond to the EARLIER full SACG training objective, not the selected no-boundary method. Simple CNN is a separate trainable refiner on FROZEN PromptDA, not PromptDA finetuning.

Before submitting, migrate into the current official journal template from https://ieeeaccess.ieee.org/author-center/, fix author information and disclosures, add independently held-out evaluations, reproduce checkpoints, measure trainable parameters and inference resources, and conduct matched-input/constant-gate ablations.

See SOURCE_MAP.md and EXPERIMENTS_TODO.md for provenance and remaining work. The manuscript was composed from existing repository source and stored evaluation JSON; no new experiments were run as part of drafting.