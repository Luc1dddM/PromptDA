# SACG working research paper

This LaTeX manuscript was written against the code and committed evaluation outputs on branch Lamns_V4_SACG. Working method: SACG with a learned LiDAR-validity field and a gate regularizer, **without** the optional boundary-weighted loss.

The manuscript is a research draft, not a submission-ready paper. It does not claim new experiments, state of the art, calibrated uncertainty, PromptDA fine-tuning, or generalization of the selected variant to ScanNet++.

Contents:
- main.tex: entrypoint, title and abstract.
- sections/: Introduction, Related Work, Method, Experiments, Results, Discussion.
- references.bib: verified academic references.
- SOURCE_MAP.md: code and result provenance.
- EXPERIMENTS_TODO.md: outstanding evidence before submission.

Compile from this directory with pdflatex main.tex, bibtex main, pdflatex main.tex, pdflatex main.tex. A generic article class is used until the target venue is selected.

Current complete quantitative evidence for the selected no-boundary configuration is limited to ARKitScenes seed 10 with 5,599 validation frames. Three-seed ARKitScenes and ScanNet++ results belong to the earlier full-objective SACG configuration; the paper labels them accordingly. The model was chosen using validation results, so a new held-out scene-level evaluation is necessary.

The full PromptDA backbone remains frozen for both SimpleRefiner and SACG. The simple CNN is **not** a fine-tuned PromptDA.
