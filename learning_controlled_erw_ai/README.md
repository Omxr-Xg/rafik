# Learning-Controlled Elephant Random Walk

This folder contains a draft research manuscript and reproducible computation
files for a Learning-Controlled Elephant Random Walk (LC-ERW), a variant of the
elephant random walk whose memory parameter is selected by a predictable online
learning rule.

## Contents

- `main.tex`: LaTeX manuscript with model definition, theorems, proofs and discussion.
- `references.bib`: BibTeX references in author-year style.
- `R/simulate_lc_erw.R`: Base R simulations for four LC-ERW policies.
- `R/check_variance.R`: Variance convergence checks across sample sizes.
- `results/`: Created automatically by the R scripts.

## Compile the Manuscript

From this folder:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses `natbib` with the `apalike` bibliography style.

## Run the Computations

From the repository root:

```sh
Rscript learning_controlled_erw_ai/R/simulate_lc_erw.R
Rscript learning_controlled_erw_ai/R/check_variance.R
```

The scripts use base R only and write CSV summaries to:

- `learning_controlled_erw_ai/results/lc_erw_simulation_summary.csv`
- `learning_controlled_erw_ai/results/lc_erw_variance_grid.csv`

## Scientific Note

The manuscript is written as a rigorous candidate contribution. It does not
claim peer-reviewed novelty or guarantee acceptance in a journal. The central
mathematical claim is conditional on the stated stable diffusive learning
assumption, and the R scripts are intended to make the variance prediction
numerically checkable.
