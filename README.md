# NeuroFaceCNN
CNN-based deep learning framework for temporally preserved face classification from naturalistic fMRI data using voxel-by-time transformations



Key Features:

- Voxel-by-Time Transformation: Converts 4D fMRI scans into 2D matrices while preserving temporal information.

- Custom CNN Model: A compact yet effective 2D CNN designed for fMRI data with spatiotemporal structure.

- Integrated Gradients Analysis: Attribution maps generated using DeepExplain to identify voxel-level contributions.

- Reconstruction in Brain Space: Neural attributions are mapped back to anatomical space and visualized with nilearn.



Dataset

Source: Naturalistic Neuroimaging Database v2.0 (NNDb)

Participants: 85 healthy adults (ages 18–58), scanned while watching full-length movies.

Labels: 3606 10-second segments labeled as "face" or "no-face" based on automated annotations.

