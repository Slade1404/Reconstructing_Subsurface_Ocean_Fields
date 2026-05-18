# MLPR Project Context: Ocean Subsurface 3D Reconstruction

## 1. Project Objective
The overarching objective of this project is to reconstruct complex 3D ocean subsurface thermodynamic profiles (Temperature and Salinity) exclusively from 2D satellite surface observations. By mapping surface phenomena to deep-water conditions, we can significantly reduce the computational cost of physical data assimilation models (like GLORYS).

## 2. Region and Datasets
- **Region**: Bay of Bengal (BoB)
- **Spatial Resolution**: 1/4° grid (81x81 map, zero-padded to 88x88 for the UNet).
- **Time Split (3 Full Years)**:
  - **Train**: Jan 1, 2023 – Dec 31, 2024 (2 years)
  - **Test**: Jan 1, 2025 – Dec 31, 2025 (1 year)

### Input Features (Surface Observations - `(88, 88, 8)`)
1. `SST` (Sea Surface Temperature)
2. `SSS` (Sea Surface Salinity)
3. `SLA` (Sea Level Anomaly)
4. `UGOS` & `VGOS` (Absolute Geostrophic Velocities)
5. `TAUX` & `TAUY` (Surface Downward Wind Stresses)
6. `CURL` (Wind Stress Curl)

### Ground Truth Targets (Subsurface - `(88, 88, 52)`)
- 26 depth levels of Temperature (`thetao`)
- 26 depth levels of Salinity (`so`)
- Depths exponentially scale from 0.49m down to ~3000m.

## 3. Physics-Informed ML Architecture
A traditional UNet treats all 52 output channels identically, often leading to physically impossible predictions (e.g., Temperature arbitrarily increasing at 2000m depth, or the 0.5m prediction completely ignoring the satellite SST).

This repository resolves this by wiring **Physics Constraints** directly into the custom training loop loss function:
- **Surface Consistency Anchor**: Hard-penalizes the model if the predicted Temperature/Salinity at the 0.5m depth layer deviates from the satellite-observed SST/SSS inputs.
- **Thermal Stability Gradient**: Penalizes the model if Temperature increases with depth, mathematically forcing the network to learn natural thermodynamic decay without blindly memorizing the training data.
