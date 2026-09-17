# Predictive Coding Networks Capture Human Neural Representations Missing in Supervised DNNs

This repository contains the code for this study:

```
Gütlin, D.*, Kittelmann, D.*, & Auksztulewicz, R. (2025). 
Predictive coding networks capture human neural representations missing in supervised DNNs.
```

Please cite it if you use it.

## Contents

**Main scripts:**
- `train_models.py` - Train neural networks with different learning objectives (predictive, contrastive, supervised)
- `run_save_model_rdms.ipynb` - Compute representational dissimilarity matrices from model activations
- `save_eeg_rdms.py` - Compute RDMs from EEG data
- `run_analysis_final.ipynb` - Main analysis notebook
- `generate_data.py` - Generate (additional) stim data

**Source code:**
- `src/training.py` - Training loop implementations
- `src/load_dataset.py` - Data loading and preprocessing utilities
- `src/load_eeg.py` - EEG data loading
- `src/analysis.py` - Analysis functions
- `src/visualization.py` - Visualization functions

**Environment:**
- `analysis_env.yml` - Conda environment for analysis
- `stats_env.yml` - Conda environment for stats