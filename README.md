# Compartment-driven Gaussian processes

Code and data for the paper on uncertainty in pharmacokinetic exposure.

Use Python 3.12. Dependencies, including SciPy 1.16.3, are pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --output /tmp/pkgp-results --jobs 3
```

The output directory must be new or empty. Alternatively, run each notebook from top to bottom.

- `simulations.ipynb`: simulation scenarios S1–S9.
- `warfarin_experiment.ipynb`: warfarin analysis.
- `theophylline_experiment.ipynb`: theophylline analysis.
- `chemo_experiment.ipynb`: 5-FU and capecitabine analyses.

The Python modules contain the shared models, inference methods, and plotting code. Inputs are in `data/`; supplied outputs are in `results/` and `figures/`. Data sources are described in `data/README.md`.
