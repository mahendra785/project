# Paper source

LNCS (MICCAI) format. Compile on Overleaf (upload this folder; select the
Springer LNCS template's `llncs.cls` + `splncs04.bst`) or locally:

    latexmk -pdf main.tex

All numbers are in `generated/`, written by
`backend/scripts/experiments/paper_macros.py` from `docs/research/results/`.
Never edit `generated/` by hand. `data/*.csv` are pgfplots-ready series
written by `aggregate.py` and `simulation.py`.
