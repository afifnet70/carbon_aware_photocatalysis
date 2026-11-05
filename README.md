# Carbon-Aware Explainable ANN Modelling of Photocatalytic Hydrogen Production

Reproducibility code for the manuscript:

**Carbon-Aware Explainable Artificial Neural Network Modelling of Photocatalytic Hydrogen Production**

The workflow follows the manuscript's protocol: 1,370 GLFO–sucrose measurements;
three operating inputs (pH, catalyst loading and initial H2O2 concentration);
a common 80/20 hold-out split; five feature sets; 23 regression algorithms;
three-fold randomized hyperparameter search; CPU-time/energy/CO2e accounting;
and five complementary explanation approaches for the selected ANN-3 model.

## Dataset access

The manuscript's Data Availability Statement says that the curated data,
selected feature subsets and training/ablation data are available from the
corresponding author on reasonable request. The experimental CSV is therefore
**not included**. Obtain the data from the corresponding author and put it at
`data/glfo_hydrogen.csv`. See `data/README.md`.

## Environment

Python 3.10+ is recommended. The manuscript reports Python 3.13 and the
libraries listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run_all.py --data data/glfo_hydrogen.csv
```

A full 23-model randomized search is computationally intensive. For a quick
pipeline check only (not manuscript reproduction):

```bash
python run_all.py --data data/glfo_hydrogen.csv --quick
```

The quick mode reduces search draws and must not be used to report the paper's
final benchmark.

Outputs are written to `results/`:
- `benchmark_results.csv`: hold-out metrics and final-fit cost for each model
- `search_cost.csv`: estimated tuning/search energy and emissions
- `ann3_model.joblib`: fitted ANN-3 pipeline and target scaler
- `holdout_predictions.csv`
- `feature_importance.csv`
- `figures/`: diagnostic, benchmark, carbon, SHAP, PDP/ICE, ALE and importance plots
- `run_metadata.json`: split seed, software versions and carbon assumptions

## Protocol notes and reproducibility boundaries

- The manuscript does not state the numerical random seed in the accessible
  methods text. The implementation exposes `--seed` and defaults to 42; set it
  to the original analysis seed if that value is available.
- The manuscript gives hyperparameter ranges and draw counts, but not every
  sampled candidate or all package-level defaults. This implementation makes
  those choices explicit in source code and records them in the output metadata.
- Energy is estimated from effective CPU time using 12 W per busy core,
  two cores, PUE 1.0 and 475 g CO2e/kWh. Memory and accelerator energy are
  excluded, as in the manuscript. Search energy is an upper-bound estimate
  from wall time with two cores assumed busy.
- Exact numerical reproduction depends on the original dataset, random seed,
  hardware, software versions and library defaults. Do not treat a rerun on
  another machine as guaranteed to reproduce every printed metric.

## Data citation

Bakır, R., Orak, C., & Bakır, H. (2026). A benchmark dataset for machine
learning-based prediction of hydrogen evolution efficiency from GLFO-catalyzed
sucrose photocatalysis. *Data in Brief, 67*, 112996.
https://doi.org/10.1016/j.dib.2026.112996

## Correspondence

Mahin Montasir Afif — afif@uttarauniversity.edu.bd
