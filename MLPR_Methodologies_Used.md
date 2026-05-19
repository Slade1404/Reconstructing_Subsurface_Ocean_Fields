# MLPR Course Methodologies Used in This Project

> **Project:** Reconstructing Subsurface Ocean Fields from Satellite Surface Observations  
> **Region:** Bay of Bengal (1/4° resolution)  
> **Temporal Scope:** 2023–2025 (3 years)

This document maps every ML technique and methodology used in this project back to the MLPR course syllabus topics, organized by pipeline stage.

---

## 1. Feature Selection and Extraction (Week 1)

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Multimodal Feature Selection** | `configs/default.yaml`, `datasets.py` | Selected 8 heterogeneous surface observation features from 5 independent satellite instruments: SST, SSS, SLA, UGOS, VGOS, TAUX, TAUY, and CURL. Each captures a different physical signal (thermal, haline, altimetric, geostrophic, wind-driven). |
| **Domain-Driven Feature Engineering** | `preprocess_wind.py` | Wind Stress Curl (∇ × τ) was **derived** from raw eastward/northward stress components using finite-difference numerical gradients on a spherical coordinate system — not directly available from satellite data. This is a physics-informed engineered feature capturing Ekman pumping dynamics. |

---

## 2. Feature Preprocessing (Week 2)

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Handling Missing Values (NaN Imputation)** | `datasets.py:121`, `evaluate_unet.py:101` | Ocean datasets contain NaN values over land pixels. All NaNs are replaced with 0.0 using `np.nan_to_num()` before feeding into the model. |
| **Z-Score Normalization (Standardization)** | `preprocess_glorys.py:50–75`, `preprocess_wind.py:68–83`, `preprocess_sst.py`, `preprocess_sss.py`, `preprocess_sla.py` | Every input feature and every target variable is Z-Score normalized: `x_norm = (x − μ) / σ`. Statistics (μ, σ) are computed **exclusively on the training split** (2023–2024) and applied to the full dataset (including 2025 test) to prevent **data leakage**. |
| **Depth-Wise Normalization** | `preprocess_glorys.py:56–64` | GLORYS Temperature and Salinity targets are normalized independently at each of the 26 depth levels, because ocean properties have vastly different statistical distributions at the surface vs. 3000m deep. |
| **Temporal Resampling** | `preprocess_wind.py:41` | Raw hourly wind stress observations are resampled to daily means using `xr.resample(time='1D').mean()` to align temporal resolution with the other daily satellite products. |
| **Spatial Regridding (Interpolation)** | `preprocess_glorys.py:41–45`, `preprocess_wind.py:64` | All datasets are bilinearly interpolated (`method='linear'`) onto a common 1/4° latitude-longitude grid (81×81) to ensure pixel-perfect spatial alignment across all 5 data sources. |
| **Train/Validation/Test Split** | `default.yaml:12–13`, `train_classical_models_bob_1_4.py:91` | Strict temporal split: **Train** = Jan 2023 – Dec 2024 (731 days), **Test** = Jan 2025 – Dec 2025 (365 days). No random shuffling — purely chronological to simulate real-world forecasting conditions and prevent temporal leakage. |
| **Zero-Padding** | `datasets.py:146–153` | The 81×81 spatial grid is zero-padded to 88×88 to satisfy the UNet's power-of-2 downsampling requirements. Padding is cropped before metric computation to avoid inflating accuracy. |

---

## 3. Classification and Regression — Classical Baselines (Weeks 5–6)

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Ridge Regression (L2-Regularized Linear Regression)** | `train_classical_models_bob_1_4.py:107` | `Ridge(alpha=1.0)` — Serves as the linear baseline. Maps 8 surface features to 52 depth channels (26 Temp + 26 Salinity) simultaneously. L2 penalty prevents coefficient explosion on correlated ocean features. |
| **Random Forest Regression** | `train_classical_models_bob_1_4.py:108` | `RandomForestRegressor(n_estimators=50, max_depth=15)` — Ensemble of 50 decision trees with depth-limited splits. Captures non-linear surface-to-subsurface relationships without requiring gradient computation. |
| **Support Vector Regression (LinearSVR)** | `train_classical_models_bob_1_4.py:109` | `MultiOutputRegressor(LinearSVR(max_iter=2000))` — Wraps 52 individual SVR models (one per depth channel) using the ε-insensitive loss function. Tests whether margin-based optimization outperforms tree-based methods. |
| **MultiOutput Regression** | `train_classical_models_bob_1_4.py:109` | `sklearn.multioutput.MultiOutputRegressor` wraps single-output models (LinearSVR) to handle the 52-dimensional target vector by training 52 independent regressors. |

---

## 4. Performance Metrics (Week 5)

| Metric | Where Used | Details |
|--------|-----------|---------|
| **RMSE (Root Mean Square Error)** | `train.py:190–191`, `evaluate_unet.py:131–132`, `train_classical_models_bob_1_4.py:129–130` | Primary regression metric. Computed per-depth-level across the entire test set to produce depth-profile curves. |
| **R² (Coefficient of Determination)** | `train.py:193–196`, `evaluate_unet.py:133–134` | Measures explained variance. Computed as `1 − SS_res / SS_tot`. Used to evaluate whether the model captures meaningful variability beyond a naive mean predictor. |
| **Train vs. Test Curves (Overfitting Detection)** | `train.py:235–257` (MetricsCallback) | Per-epoch RMSE and R² are computed on both train and test sets, then plotted side-by-side for 10 strategic depth levels. Divergence between curves signals overfitting. |
| **Depth-Profile Comparative Plots** | `evaluate_unet.py:160–198` | RMSE and R² are plotted against ocean depth (inverted y-axis) for all models simultaneously, enabling direct visual comparison of UNet vs. Ridge vs. RandomForest vs. LinearSVR at every depth level. |

---

## 5. Challenges of ML Systems (Week 7)

| Challenge | How Addressed | Details |
|-----------|--------------|---------|
| **Overfitting** | L2 Weight Decay, Dropout, Physics Constraints, Train/Test monitoring | Multiple regularization strategies combined; per-epoch train-vs-test gap monitoring detects divergence early. |
| **Insufficient Data** | 3-year temporal expansion (2023–2025) | Expanded from 1 year to 3 years of daily satellite data, increasing training samples from ~365 to ~731 daily maps. |
| **Poor Quality Data / Missing Values** | NaN masking, valid-pixel filtering | Land pixels and sensor gaps produce NaN values. These are systematically handled via `nan_to_num` imputation and `valid_mask` filtering in classical models. |
| **Noise in Dataset** | Z-Score normalization, BatchNorm, Weight Decay | Normalization stabilizes input distributions; BatchNorm handles internal covariate shift; L2 decay prevents the model from fitting to high-frequency noise. |
| **Data Leakage Prevention** | Train-only normalization statistics | All μ and σ values are computed exclusively on 2023–2024 data. The 2025 test set is normalized using those frozen statistics, ensuring the model never "sees" future information. |

---

## 6. Neural Networks (Weeks 10–11)

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Multilayer Neural Network** | `unet2d.py` | The 2D-UNet is a deep multilayer network with an encoder-decoder architecture containing ~2.5M trainable parameters across 4 encoder stages, a bottleneck, and 4 decoder stages. |
| **Backpropagation** | `train.py:68–75` | Custom `train_step` uses `tf.GradientTape()` to compute gradients of the composite loss function and applies them via `optimizer.apply_gradients()`. |
| **Adam Optimizer (Adaptive Learning Rate)** | `train.py:315–316` | `tf.keras.optimizers.Adam(lr=0.001)` — Combines momentum and RMSProp for adaptive per-parameter learning rates. This is directly from Week 11's "Accelerated Learning" topic. |
| **Learning Rate = 0.001** | `default.yaml:28` | Standard initial learning rate for Adam, balancing convergence speed and stability. |
| **He Normal Weight Initialization** | `unet2d.py:15,22` | `kernel_initializer="he_normal"` — Specifically designed for ReLU activations to prevent vanishing/exploding gradients in deep networks (Week 11 topic). |
| **L2 Regularization (Weight Decay)** | `unet2d.py:13,38`, `default.yaml:24` | `kernel_regularizer=l2(0.001)` applied to every Conv2D layer. Penalizes large weights to prevent overfitting — directly from Week 11's regularization topic. Final value selected via Phase 1 grid search. |
| **Dropout Regularization** | `unet2d.py:19–20` | `layers.Dropout(dropout_rate)` applied after each conv block. Randomly zeros activations during training to prevent co-adaptation. Tuned via grid search; optimal value was 0.0 (the model benefited more from L2 alone). |
| **Batch Normalization** | `unet2d.py:17,24` | `layers.BatchNormalization()` after every Conv2D layer. Normalizes intermediate activations to stabilize training and allow higher learning rates — directly from Week 11. |
| **ReLU Activation Function** | `unet2d.py:18,25` | `layers.Activation("relu")` — Standard non-linearity preventing vanishing gradients in deep networks. |
| **Linear Output Activation** | `unet2d.py:74` | `activation="linear"` on the final 1×1 Conv2D — appropriate for regression tasks where outputs are continuous Z-Score normalized values (not bounded to [0,1]). |

---

## 7. Deep Learning — Convolutional Neural Networks (Week 12)

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Convolutional Neural Network (CNN)** | `unet2d.py` | The entire model is built from 2D convolutional layers that exploit spatial correlations in ocean surface maps. |
| **Weight Sharing** | Inherent in Conv2D | Each convolutional filter slides across the entire spatial map, sharing weights across all positions — fundamental CNN property. |
| **Convolution (3×3 kernels)** | `unet2d.py:15,22` | `Conv2D(filters, 3, padding="same")` — 3×3 convolutions with "same" padding preserve spatial dimensions while learning local spatial patterns. |
| **Padding ("same")** | `unet2d.py:15,22` | Zero-padding applied to maintain spatial dimensions through convolution, preventing progressive shrinkage of feature maps. |
| **Pooling (MaxPool2D)** | `unet2d.py:32` | `MaxPool2D(2)` — 2×2 max pooling halves spatial dimensions at each encoder stage, creating a hierarchical multi-scale representation. |
| **Transposed Convolution (Upsampling)** | `unet2d.py:39` | `Conv2DTranspose(filters, 2, strides=2)` — Learned upsampling that doubles spatial dimensions in the decoder, reconstructing fine-grained spatial detail. |
| **Skip Connections (U-Net Architecture)** | `unet2d.py:41,69–71` | `layers.concatenate([x, conv_features])` — Encoder features are concatenated with decoder features at matching resolutions, allowing the network to recover fine spatial details lost during downsampling. This is the defining feature of the U-Net architecture. |
| **Encoder-Decoder Architecture** | `unet2d.py:60–71` | 3-stage encoder (progressive downsampling) → bottleneck → 3-stage decoder (progressive upsampling). Learns both global context and local detail. |
| **Multi-Scale Feature Hierarchy** | `default.yaml:22` | `filters: [32, 64, 128, 256]` — Filter count doubles at each encoder stage, creating increasingly abstract representations from local surface patterns to basin-scale ocean dynamics. |

---

## 8. Hyperparameter Tuning (Week 11)

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Grid Search (Phase 1 — Regularization)** | `tune_unet.py:31–36` | Exhaustive 3×3 grid over `dropout_rate ∈ {0.0, 0.1, 0.2}` and `weight_decay ∈ {0.0, 1e-4, 1e-3}` = 27 total runs. Selected optimal: `dropout=0.0, weight_decay=0.001`. |
| **Grid Search (Phase 2 — Physics Loss Weights)** | `tune_unet.py:37–42` | Targeted 4×3 grid over `lambda_stability ∈ {0.0, 0.05, 0.1, 0.5}` and `lambda_surface ∈ {0.5, 1.0, 2.0}` = 12 runs. Selected optimal: `stability=0.05, surface=0.5`. |
| **Fast-Mode Training (Reduced Epochs for Tuning)** | `tune_unet.py:26,55` | `TUNE_EPOCHS = 5` during grid search to rapidly evaluate combinations before committing to a full 50-epoch run. Checkpointing and backup disabled to maximize speed. |
| **Warm-Start / Resume Training** | `train.py:140–152` | Metrics history is loaded from disk on startup and appended to, enabling pause/resume without losing progress. |

---

## 9. Custom Loss Functions — Physics-Informed Constraints

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Multi-Task Loss (Composite Loss Function)** | `train.py:63` | `total_loss = dense_loss + λ_surface × surface_loss + λ_stability × stability_loss` — Three loss terms are combined with tunable weight coefficients. |
| **Dense MSE Loss (Primary Reconstruction)** | `train.py:42` | `tf.reduce_mean(tf.square(y_true − y_pred))` — Standard Mean Squared Error across all 52 output channels (26 Temp + 26 Salinity at all depths). |
| **Surface Consistency Anchor (SST)** | `train.py:47–49` | Penalizes deviation between the predicted Temperature at 0.5m depth (`y_pred[..., 0:1]`) and the satellite SST input (`x[..., 0:1]`). Forces the model to respect known surface observations. |
| **Surface Consistency Anchor (SSS)** | `train.py:51–53` | Penalizes deviation between the predicted Salinity at 0.5m depth (`y_pred[..., 26:27]`) and the satellite SSS input (`x[..., 1:2]`). Dual-anchoring for both thermodynamic variables. |
| **Thermal Stability Gradient Constraint** | `train.py:59–61` | `tf.nn.relu(temp_pred[..., 1:] − temp_pred[..., :-1])` — Penalizes physically impossible temperature inversions (temperature increasing with depth). Only positive gradients are penalized via ReLU, allowing natural thermoclines. |

---

## 10. Data Pipeline and Engineering

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **tf.data.Dataset Generator Pipeline** | `datasets.py:162–191` | Lazy data loading via Python generators wrapped in `tf.data.Dataset.from_generator()`. Avoids loading the entire 15GB+ dataset into RAM. |
| **Batch Processing** | `datasets.py:191`, `default.yaml:10` | `.batch(16)` groups 16 samples per training step, balancing GPU memory usage and gradient estimation quality (Mini-batch from Week 11). |
| **Data Prefetching** | `datasets.py:191` | `.prefetch(tf.data.AUTOTUNE)` overlaps data loading with model computation, eliminating I/O bottlenecks. |
| **Random Temporal Sampling** | `datasets.py:99` | `np.random.choice(self.common_times)` randomly samples timesteps each epoch, providing implicit data augmentation through varied temporal ordering. |
| **Checkpoint Saving** | `train.py:331–333` | `ModelCheckpoint(save_weights_only=True)` saves model weights after every epoch, enabling recovery from crashes and enabling post-training evaluation. |
| **BackupAndRestore** | `train.py:334` | `BackupAndRestore(backup_dir)` — Keras callback that saves full optimizer state, enabling seamless mid-training resume. |

---

## 11. Additional Techniques Not Explicitly in Syllabus (But Used)

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Physics-Informed Neural Networks (PINNs)** | `train.py:40–64` | The custom loss function injects domain-specific physical constraints (surface anchoring and thermal stability) directly into the gradient computation. This is an active research area beyond standard ML curricula. |
| **Multi-Output Regression** | Entire pipeline | The model simultaneously predicts 52 continuous variables (26 depths × 2 variables) from 8 inputs — a high-dimensional regression problem. |
| **Spatial Padding and Cropping for FCNs** | `datasets.py:146–153`, `train.py:168–172` | Zero-padding inputs to satisfy architectural constraints (divisibility by 2^N for N pooling layers), then cropping predictions before metric computation to avoid artificial accuracy inflation. |
| **Feature Stacking (Channel Concatenation)** | `datasets.py:121` | 8 heterogeneous 2D surface observation maps are stacked along the channel dimension to form a single `(H, W, 8)` tensor — analogous to RGB channels in computer vision but with 8 physically meaningful channels. |
| **Bilinear Spatial Interpolation (Regridding)** | All `preprocess_*.py` scripts | Aligning datasets from different satellite instruments onto a common spatial grid using `xr.interp(method='linear')`. |
| **Derived Feature Engineering (Wind Stress Curl)** | `preprocess_wind.py:7–17` | Computing ∇×τ from raw wind stress vectors using numerical differentiation on a spherical coordinate system with Earth radius correction. |

---

## Summary: Course Week Coverage

| Week | Topic | Coverage in Project |
|------|-------|-------------------|
| 1 | What is ML, Types of ML, Feature Selection/Extraction | ✅ Multimodal satellite feature selection, supervised regression |
| 2 | Preprocessing, Train/Val/Test splits | ✅ Z-Score normalization, NaN handling, temporal splitting, data leakage prevention |
| 5 | Regression, Performance Metrics | ✅ RMSE, R², per-depth evaluation, train-vs-test curves |
| 6 | SVMs, SVR, Ensemble Methods | ✅ LinearSVR, Ridge Regression, Random Forest baselines |
| 7 | Challenges of ML (overfitting, noise, data quality) | ✅ Regularization strategies, data quality handling, overfitting monitoring |
| 10 | Neural Networks, Backpropagation | ✅ Custom `train_step` with `GradientTape`, multi-layer architecture |
| 11 | Hyperparameters, Learning Rate, Regularization, BatchNorm | ✅ Adam optimizer, L2 decay, Dropout, BatchNorm, He init, 2-phase grid search |
| 12 | CNNs, Convolution, Pooling, Architectures, Weight Sharing | ✅ Full 2D-UNet with skip connections, MaxPool, Conv2DTranspose, multi-scale filters |
