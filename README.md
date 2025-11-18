# Pycytominer_Validation

Pycytominer-based profile quality evaluation for Cell Painting embeddings.

This repository provides small, model-agnostic scripts to compute standard
morphological profiling metrics:

- **Percent Replicating (PR)**
- **Percent Matching (PM)**

from per-well embedding tables (e.g. `well_features.csv`) produced by models
such as MOAProfiler or CellPaintSSL.

## Scripts

- `validate_metrics.py`  
  Main entry point: compute PR and PM with a selectable similarity metric
  (`pearson`, `spearman`, `cosine`). Uses `stat_helpers_metrics.py` internally.
  The script builds null distributions from non-replicate / non-match groups
  and reports the 95th percentile threshold and PR/PM values.

- `stat_helpers_metrics.py`  
  Helper functions to:
  - compute median pairwise similarity within each group,
  - construct null distributions from non-replicate / non-match groups,
  - derive the 95th percentile threshold and PR/PM.

- `graph_helpers.py`  
  Helper functions for plotting the null distribution (KDE) and per-group
  median similarities with the threshold.

## Dependencies

The `requirements.txt` file specifies a full profiling environment including
Jupyter, plotting libraries, and pinned versions of `pycytominer` and
`cytominer-eval`:

```bash
pip install -r requirements.txt
```
A minimal setup for running only the validation scripts would require:
numpy, pandas, scipy, matplotlib, and seaborn, but the provided
requirements.txt matches the broader environment used in our analyses.

Typical usage

Assuming a CSV file `well_features_normalized.csv` with columns like:

batch, plate, well

perturbation_id (compound)

target (MOA / biological target)

emb... (embedding columns)

run:

# Percent Replicating / Matching with Pearson correlation (default)
```
python validate_metrics.py /path/to/well_features_normalized.csv
```

# Same, but using cosine similarity
```
python validate_metrics.py /path/to/well_features_normalized.csv --metric cosine
```
The script creates timestamped `reports/<timestamp>_<METRIC>/` folders with:

`summary_report.txt` (PR/PM values),

CSVs with per-group median similarities,

CSVs with null distributions,

PNG plots of null distributions and per-group median similarities.
## License

This project is licensed under the Apache License, Version 2.0.
See the `LICENSE` file for details.

