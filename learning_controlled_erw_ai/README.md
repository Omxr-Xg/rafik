# Learning-Controlled Elephant Random Walk

This folder contains a draft research manuscript and reproducible computation
files for a Learning-Controlled Elephant Random Walk (LC-ERW), a variant of the
elephant random walk whose memory parameter is selected by a predictable online
learning rule.

## Contents

- `main.tex`: LaTeX manuscript with model definition, theorems, proofs and discussion.
- `references.bib`: BibTeX references in author-year style.
- `R/simulate_lc_erw.R`: Base R simulations for four LC-ERW policies, including QQ-plots.
- `R/check_variance.R`: Variance convergence checks across sample sizes with confidence intervals.
- `python/simulate_lc_erw.py`: Python fallback used to generate the included numerical results when `Rscript` is unavailable.
- `requirements.txt`: Python dependencies for the fallback simulation pipeline.
- `results/`: Created automatically by the simulation scripts.

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

If `Rscript` is not installed, run the Python fallback:

```sh
python3 learning_controlled_erw_ai/python/simulate_lc_erw.py
```

The R scripts use base R only. The Python fallback requires `numpy`, `scipy`,
and `matplotlib`; install them with:

```sh
python3 -m pip install -r learning_controlled_erw_ai/requirements.txt
```

The current numerical tables, QQ-plots, path diagnostics and scaling plots in
the manuscript were generated with the Python fallback.

The scripts write CSV summaries to:

- `learning_controlled_erw_ai/results/lc_erw_simulation_summary.csv`
- `learning_controlled_erw_ai/results/lc_erw_variance_grid.csv`
- `learning_controlled_erw_ai/results/lc_erw_simulation_summary_python.csv`
- `learning_controlled_erw_ai/results/lc_erw_variance_grid_python.csv`
- `learning_controlled_erw_ai/results/fclt_covariance_python.csv`
- `learning_controlled_erw_ai/results/extended_diagnostics_python.csv`

The default simulation scripts also write plot PDFs to
`learning_controlled_erw_ai/results/`, including QQ-plots, histogram overlays,
`variance_ratio_grid.pdf`, learned-policy trajectories, critical scaling and
superdiffusive scaling diagnostics.

## Scientific Note

The manuscript is written as a rigorous candidate contribution. It does not
claim peer-reviewed novelty or guarantee acceptance in a journal. The central
mathematical claim is conditional on the stated stable diffusive learning
assumption, and the R scripts are intended to make the variance prediction
numerically checkable.
