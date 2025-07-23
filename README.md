# Functional Connectivity — Reproducible rs‑fMRI Pipeline  
Analysis of functionally integrated relationships between spatially separated
brain regions
------------------------------------------------------------------------------

This repository accompanies **“Exploring human brain‑development patterns on
resting‑state networks using graph theory and machine learning”**
(Arsalidou *et al.*, 2025).  
It contains **all scripts, configuration files and Docker resources** needed to
reproduce the results reported in the manuscript.

> **TL;DR for reviewers**  
> 1. `git clone https://github.com/EtokonE/functional_connectivity.git`  
> 2. Pull the ready‑made Docker image `etokone/functional_connectivity:latest`  
> 3. Mount your BIDS‑formatted dataset and run the four pipeline stages  
> 4. Find the outputs in `results/`

---

## Table of contents
1. [Quick start](#quick-start)
2. [Prerequisites](#prerequisites)  
   2.1 [rs‑fMRI preprocessing (fMRIPrep 22.0.2)](#rs-fmri-preprocessing-fmriprep-2202)
3. [Dataset layout](#dataset-layout)
4. [Pipeline overview](#pipeline-overview)  
   4.1 [Functional‑connectivity metrics (raw feature space)](#functional-connectivity-metrics-raw-feature-space)
5. [Module structure](#module-structure)  
   5.1 [Graph‑theoretical feature space](#graph-theoretical-feature-space)
6. [Configuration](#configuration)  
   6.1 [Data preparation & feature engineering](#data-preparation--feature-engineering)
7. [Expected outputs](#expected-outputs)
8. [Machine‑learning workflow](#machine-learning-workflow)
9. [Re‑running the ML benchmarks](#re-running-the-ml-benchmarks)
10. [Where to go next](#where-to-go-next)
11. [Citing this work](#citing-this-work)
12. [License](#license)

---

## Quick start

```bash
# 1. Clone & enter the repo
git clone https://github.com/EtokonE/functional_connectivity.git
cd functional_connectivity

# 2. Pull the reproducible environment
docker pull etokone/functional_connectivity:latest

# 3. Launch an interactive JupyterLab inside the container
docker run -it --rm -p 8888:8888 \
  -v /ABS/PATH/TO/functional_connectivity:/home/neuro/functional_connectivity \
  -v /ABS/PATH/TO/BIDS_DATASET:/data \
  etokone/functional_connectivity jupyter lab
````

The container already satisfies **Python 3.7** and every package listed in
`requirements.txt`. It also exposes the code as a Python module so the
scripts can be called from notebooks or the command line.

---

## Prerequisites

| Item                        | Version / notes                                                                                         |
| --------------------------- |---------------------------------------------------------------------------------------------------------|
| **Python**                  | ≥ 3.7 (3.7 used in Docker)                                                                             |
| **BIDS‑compatible dataset** | One folder per age group (`adults/`, `teenagers/`, `yong_children/`) containing pre‑processed fMRI runs |
| **fMRIPrep outputs**        | Pre‑whitened & MNI‑152‑normalised BOLD images                                                           |
| Optional                    | GPU is *not* required; everything runs on CPU                                                           |

### rs‑fMRI preprocessing (fMRIPrep 22.0.2)

| Step | Anatomical (T1w)                                                | Functional (BOLD)                                                                                |
| ---- |-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1    | N4 bias‑field correction                                        | Reference volume & skull‑strip                                                                   |
| 2    | Skull‑strip (`antsBrainExtraction`)                             | Slice‑time correction (`3dTshift`)                                                               |
| 3    | FAST tissue segmentation (CSF/WM/GM)                            | Head‑motion realignment (6 DoF)                                                                  |
| 4    | Surface reconstruction (FreeSurfer 7.4)                         | Susceptibility‑distortion correction (field‑map based)                                           |
| 5    | Spatial normalisation → **MNI‑ICBM152 2009c** (non‑linear ANTs) | BBR T1w‑to‑BOLD co‑registration                                                                  |
| 6    | —                                                               | Confound regression: 6 motion + derivatives, CSF/WM/Global, aCompCor‑5, *FD* & *DVARS* censoring |
| 7    | —                                                               | Temporal filtering 0.01–0.1 Hz (after nuisance removal)                                          |
| 8    | —                                                               | Output: pre‑whitened, MNI‑aligned 4 D NIfTI in `derivatives/fmriprep/sub-*/func/`                |

---

## Dataset layout

Edit two variables in `src/config/base_config.py` before running:

```python
_C.PATH.ROOT = '/absolute/path/to/functional_connectivity'
_C.PATH.DATA = '/absolute/path/to/BIDS_DATASET'
```

Inside `PATH.DATA` we expect:

```
.
├── adults/
│   └── sub-*/ses-*/func/sub-*_task-rest_space-MNI152NLin2009cAsym_bold.nii.gz
├── teenagers/
└── yong_children/
```

If your folder names differ, change

```
_C.BIDS.ADULTS_BIDS_ROOT
_C.BIDS.TEENAGERS_BIDS_ROOT
_C.BIDS.CHILDREN_BIDS_ROOT
```

and any other relevant keys in the config.

---

## Pipeline overview

| Stage                   | Script (call order)                                                      | Key output                                                           | Source code             |
| ----------------------- |--------------------------------------------------------------------------|----------------------------------------------------------------------|-------------------------|
| **1. Parcellation**     | `python src/parcellation.py`                                             | `results/parcellation/*/parcellation.h5` (100‑parcel Schaefer atlas) | `src/parcellation.py`   |
| **2. Connectivity**     | `python src/fc_measure.py`                                               | Eleven connectivity matrices per subject (`pearson`, `coherence`, …) | `src/fc_measure.py`     |
| **3. Graph metrics**    | `python src/graph_analysis.py`                                           | 5 local + 9 global graph metrics per connectivity type               | `src/graph_analysis.py` |
| **4. Machine learning** | `python src/machine_learning/raw_connectivity_prediction.py` (and peers) | Cross‑validated SVC accuracy, selected features, weight maps         | `src/machine_learning/` |

Each step reads its predecessor’s `.h5` output and writes a new one; paths are
assembled automatically with `define_h5_path()`.

### Functional‑connectivity metrics (raw feature space)

The pipeline implements **11 complementary measures** that capture linear / non‑linear (dis)similarities across *time*, *frequency* and *time–frequency* domains.
All metrics are pair‑wise for the 100‑parcel atlas and saved as dense, symmetric
*n* × *n* matrices (`float32`, Fisher‑*z* or 0‑1 scaled where applicable).

| Domain →   | Time        | Time                  | Time        | Time       | Time                | Frequency  | Time‑frequency    | Time                  | Time            | Time             | Time                    |
|------------|-------------|-----------------------|-------------|------------|---------------------| ---------- | ----------------- |-----------------------| --------------- | ---------------- |-------------------------|
| Metric ↓   | Pearson *r* | Cross‑corr. (max lag) | Partial *r* | Spearman ρ | Percentage‑bend *r* | Coherence  | Wavelet coherence | Mutual information    | Euclidean dist. | City‑block dist. | Dynamic time warping    |
| Type       | similarity  | similarity            | similarity  | similarity | similarity (robust) | similarity | similarity        | similarity (non‑lin.) | dissimilarity   | dissimilarity    | dissimilarity (elastic) |
| Linear?    | ✔︎          | ✔︎                    | ✔︎          | ✖︎         | ✖︎                  | ✔︎         | ✔︎                | ✖︎                    | ✔︎              | ✔︎               | ✖︎                      |
| Scale‑inv. | ✔︎          | ✔︎                    | ✔︎          | ✔︎         | ✔︎                  | ✔︎         | ✔︎                | ✖︎                    | ✖︎              | ✖︎               | ✖︎                      |


---

## Module structure

```
src/
├── config/              # yacs‑based hierarchical config
├── parcellation.py      # atlas‑based time‑series extraction
├── fc_measure.py        # 11 connectivity estimators
├── graph_analysis.py    # local & global network metrics
├── machine_learning/
│   ├── raw_connectivity_prediction.py
│   ├── graph_connectivity_prediction.py
│   └── combined_connectivity_prediction.py
└── utils.py, logger.py  # small helpers
resources/rois/          # Schaefer 2018 100‑parcel atlas
```

### Graph‑theoretical feature space

For every thresholded FC matrix an **undirected weighted graph** is built
with `networkx`.
We extract **14 metrics**—five local (node‑wise) and nine global—yielding
100 × 5 + 9 = 509 features per connectivity definition.

| Category   | Metric                        | Intuition                                 |
| ---------- | ----------------------------- |-------------------------------------------|
| **Local**  | Degree centrality             | number of edges → integration potential   |
|            | *Avg.* neighbour degree       | “lead‑a‑rich‑club” tendency               |
|            | Betweenness centrality        | bridge that controls shortest paths       |
|            | Closeness centrality          | inverse farness from all others           |
|            | Clustering coefficient        | local segregation / cliquishness          |
| **Global** | Assortativity                 | like‑with‑like wiring; resilience         |
|            | *Avg.* clustering coefficient | mean local segregation                    |
|            | *Avg.* shortest‑path length   | global efficiency                         |
|            | Density                       | wiring cost                               |
|            | Cost‑efficiency               | density ↔︎ efficiency trade‑off           |
|            | Transitivity                  | triangle prevalence                       |
|            | Radius & diameter             | graph compactness limits                  |
|            | Small‑worldness               | optimal segregation + integration balance |


---

## Configuration

All hyper‑parameters (filter bands, TR, feature‑selection *k*, SVC *C*,
thresholding strategy, etc.) live in `src/config/base_config.py`.
**To reproduce the manuscript**, run the pipeline with **no external YAML
override**—the defaults already match the paper.

If you wish to explore alternatives, pass a YAML path to `combine_config()`
(see the docstring at the end of the config file).

### Data preparation & feature engineering

1. **Atlas time‑series extraction** — `nilearn.maskers.NiftiLabelsMasker`
   with 0.01–0.1 Hz band‑pass, first‑order detrend and CompCor noise
   regressors.
2. **Feature matrices** — 11 FC matrices × subjects → compressed `HDF5`
   (`h5py`).
3. **Graph transformation** — helper `build_graph()` returns
   `networkx.Graph` and a NumPy feature vector.
4. **Concatenation strategies**

   * *Raw* : upper‑triangle of a single FC matrix (4950 dims).
   * *Graph* : 509‑length vector per FC metric.
   * *Intra‑domain* : `[raw ‖ graph]` within the **same** metric.
   * *Cross‑domain* : stack of all intra‑domain vectors (≈ 61 k dims).
5. **Z‑scoring** — features standardised on the training fold inside
   cross‑validation.

---

## Expected outputs

```
results/
├── parcellation/Schaefer2018_100Parcels_7Networks/
│   ├── adults_parcellation.h5
│   ├── teenagers_parcellation.h5
│   └── yong_children_parcellation.h5
├── connectivity/Schaefer2018_100Parcels_7Networks/
│   ├── adults_connectivity.h5
│   ├── teenagers_connectivity.h5
│   └── yong_children_connectivity.h5
├── graph_features/Schaefer2018_100Parcels_7Networks/
│   ├── adults_<metric>_{global|local}_graph_features.h5
│   ├── teenagers_<metric>_{global|local}_graph_features.h5
│   └── young_children_<metric>_{global|local}_graph_features.h5
└── ml/
     ├── raw_connectivity/
     │   ├── <metric>.json
     │   └── <metric>_sfs.png
     ├── graph_connectivity/
     │   ├── <metric>.json
     │   └── <metric>_sfs.png
     └── combined_connectivity/
         ├── <metric>.json
         └── <metric>_sfs.png
```

---

## Machine‑learning workflow

* **Estimator**   : `sklearn.svm.SVC(kernel='linear')`
* **Wrapper FS**  : *Sequential Forward Selection* (SFS) with *k* ≤ 50,
  `floating=False`, `cv=5`.
* **Outer CV**   : stratified 5‑fold, repeated 20 × with different seeds to
  estimate µ ± σ accuracy.
* **Model selection**: best fold‑averaged accuracy; statistical comparison via
  Welch’s *t* and Benjamini‑Hochberg FDR.
* **Feature importances**: absolute, ℓ₂‑normalised SVC coefficients
  re‑estimated on the full training split.

---

## Re‑running the ML benchmarks

```bash
python src/machine_learning/raw_connectivity_prediction.py \
       --age-groups adults teenagers yong_children \
       --output-dir results/ml/raw_connectivity
```

Equivalent scripts exist for the **graph** and **combined** feature spaces:

```bash
python src/machine_learning/graph_connectivity_prediction.py
python src/machine_learning/combined_connectivity_prediction.py
```

All three scripts share the same config, so any preprocessing change is
reflected everywhere.

---

## Citing this work

If you use any part of this code or derive new analyses from it, **please
cite**

> *Arsalidou, M., Kalinin, M., & Faber, A. (2025).*
> *Exploring human brain‑development patterns on resting‑state networks using
> graph theory and machine learning.*
> *NeuroImage, ###, XXX‑XXX.*

```bibtex
@article{Arsalidou2025FC,
  title   = {Exploring human brain development patterns on resting-state networks
             using graph theory and machine learning},
  author  = {Arsalidou, Marie and Kalinin, Maxim and Faber, Andrei},
  journal = {NeuroImage},
  year    = {2025}
}
```

---

## License

Unless stated otherwise in individual files, **MIT License** applies.
See `LICENSE` for full text.

```
MIT © 2025 EtokonE & contributors
```

