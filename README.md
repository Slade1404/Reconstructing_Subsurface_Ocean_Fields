# Reconstructing Subsurface Ocean Fields

This repository contains a full Machine Learning pipeline designed to reconstruct 3D subsurface ocean thermodynamics (Temperature and Salinity up to 3000m deep) exclusively from 2D satellite surface observations using a custom physics-constrained 2D-UNet.

## Pipeline Overview

1. **`download_scripts/`**: Interfaces with the Copernicus Marine Environment Monitoring Service (CMEMS) MOTU APIs to download high-resolution (1/4°) daily satellite readings for Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Level Anomaly (SLA), and Geostrophic Winds.
2. **`preprocess_bob_1_4/`**: Aligns all satellite inputs to a strict spatial grid for the Bay of Bengal, calculates Wind Stress Curl, and strictly applies Z-Score normalization utilizing only the training temporal slice to prevent data leakage.
3. **`2dUNet/`**: The core ML architecture built in TensorFlow/Keras. Includes a custom training loop that integrates multi-task physics constraints (Surface Consistency anchors and Vertical Thermal Stability gradients).

## Results
The trained model drastically outperforms classical ML baselines (RandomForest, LinearSVR, Ridge) and proves that enforcing thermodynamic physics constraints significantly accelerates convergence and improves R² generalization on unseen temporal slices.

See `2dUNet/results/` for full evaluation metrics and comparative depth profiles.
