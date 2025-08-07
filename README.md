# NeuroFaceCNN
CNN-based deep learning framework for temporally preserved face classification from naturalistic fMRI data using voxel-by-time transformations

## Overview

**NeuroFaceCNN** is a deep learning framework for decoding face-related brain activity using naturalistic fMRI data. It transforms high-dimensional 4D brain volumes into compact 2D voxel-by-time matrices to preserve temporal dynamics and simplify spatial complexity. The approach enables efficient CNN-based classification of 10-second fMRI windows as “face” or “no-face” events and employs attribution methods (Integrated Gradients via DeepExplain) to reconstruct interpretable neural representations in anatomical brain space.

This repository supports the manuscript:

**Asadi, S. et al. (2025)** *Computationally Efficient Deep Learning for Temporally Preserved Face Classification in Naturalistic fMRI*


Key Features:

- Voxel-by-Time Transformation: Converts 4D fMRI scans into 2D matrices while preserving temporal information.

- Custom CNN Model: A compact yet effective 2D CNN designed for fMRI data with spatiotemporal structure.

- Integrated Gradients Analysis: Attribution maps generated using DeepExplain to identify voxel-level contributions.

- Reconstruction in Brain Space: Neural attributions are mapped back to anatomical space and visualized with nilearn.



Dataset

Source: Naturalistic Neuroimaging Database v2.0 (Aliko et al., 2020)

Participants: 85 healthy adults (42F, ages 18–58)

Stimuli: Movie-watching fMRI with detailed face event annotations

Labels: 3606 10-second segments labeled as "face" or "no-face" 


## Repository Structure

| File | Description |
|------|-------------|
| `1-preprocess_naturalistic_fmri.py` | Creates voxel-by-time matrices from raw 4D fMRI data |
| `2-load-data.py` | Loads and labels segmented fMRI samples |
| `3-train_voxel_time_cnn.py` | Trains CNN model on voxel-by-time input |
| `4-integrated_gradients.py` | Computes voxel-level IG attribution maps |
| `5-revert-IG-tobrain.py` | Maps 2D IG attributions back to brain space |
| `6-average_IG_per_subject.py` | Aggregates and averages IG maps by subject |
| `7-visualize_IG_surface_maps.py` | Projects IG maps onto fsaverage cortical surface |
| `README.md` | Project overview and usage |

## Setup

This codebase requires Python 3.8+ and the following libraries:

- `tensorflow>=2.10`
- `numpy`, `matplotlib`, `nibabel`, `nilearn`, `scikit-learn`
- `DeepExplain` (for attribution)
- `nibabel`, `scipy`, `pandas`

Mixed precision training and GPU acceleration (e.g., RTX 3090) are recommended.


## Usage

1. **Preprocess fMRI Data**
```bash
python 1-preprocess_naturalistic_fmri.py
```

2. **Load Data and Labels**
```bash
python 2-load-data.py
```

3. **Train CNN Model**
```bash
python 3-train_voxel_time_cnn.py
```

4. **Run Integrated Gradients Attribution**
```bash
python 4-integrated_gradients.py
```

5. **Reconstruct Brain-Space IG Maps**
```bash
python 5-revert-IG-tobrain.py
```

6. **Average Maps Across Time**
```bash
python 6-average_IG_per_subject.py
```

7. **Visualize Surface Projections**
```bash
python 7-visualize_IG_surface_maps.py
```



## Citation

If you use this code, please cite:

Asadi, S. et al. (2025). *Computationally Efficient Deep Learning for Temporally Preserved Face Classification in Naturalistic fMRI*.



