"""
Post-training evaluation script for the 2D-UNet.

Loads saved weights, runs inference on the test set, computes per-depth
RMSE and R², and generates comparative depth profile plots overlaid with
classical model baselines (Ridge, RandomForest, LinearSVR).
"""
import os
import json
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import warnings


def main(config_path="configs/default.yaml"):
    warnings.filterwarnings('ignore')

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_name = config['model']['name']
    results_dir = config['training']['results_dir']
    checkpoint_path = os.path.join(
        config['training']['checkpoint_dir'], "model_latest.weights.h5")

    print(f"=== {model_name} — Post-Training Evaluation ===")

    # ------------------------------------------------------------------
    # 1. Load test-split data
    # ------------------------------------------------------------------
    print("Loading test data...")
    ds_sst = xr.open_dataset(data_cfg['sst_path']).sel(time=slice(*data_cfg['time_test']))
    ds_sss = xr.open_dataset(data_cfg['sss_path']).sel(time=slice(*data_cfg['time_test']))
    ds_sla = xr.open_dataset(data_cfg['sla_uv_path']).sel(time=slice(*data_cfg['time_test']))
    ds_wind = xr.open_dataset(data_cfg['wind_path']).sel(time=slice(*data_cfg['time_test']))
    ds_glorys = xr.open_dataset(data_cfg['glorys_path']).sel(time=slice(*data_cfg['time_test']))

    # Auto-detect variable names
    sla_var = 'sla' if 'sla' in ds_sla else 'adt'
    temp_var = 'thetao_norm' if 'thetao_norm' in ds_glorys else 'thetao'
    salt_var = 'so_norm' if 'so_norm' in ds_glorys else 'so'

    depths = ds_glorys.depth.values
    n_depths = len(depths)
    patch_size = data_cfg['patch_size']
    lat_size = ds_sst.sizes['latitude']
    lon_size = ds_sst.sizes['longitude']
    pad_h = patch_size - lat_size
    pad_w = patch_size - lon_size

    common_times = sorted(
        set(ds_sst.time.values) & set(ds_sss.time.values) &
        set(ds_sla.time.values) & set(ds_wind.time.values) & set(ds_glorys.time.values))
    print(f"  Test days: {len(common_times)}, Depths: {n_depths}, "
          f"Spatial: {lat_size}x{lon_size} (padded to {patch_size}x{patch_size})")

    # ------------------------------------------------------------------
    # 2. Build model and load weights
    # ------------------------------------------------------------------
    print(f"Loading weights from {checkpoint_path}...")
    from src.models.unet2d import build_unet2d
    from src.train import OceanUNet
    base_model = build_unet2d(
        input_shape=(patch_size, patch_size, config['model']['in_channels']),
        out_channels=config['model']['out_channels'],
        filters=config['model']['filters'],
        dropout_rate=config['model'].get('dropout_rate', 0.0),
        weight_decay=config['model'].get('weight_decay', 0.0))
    wrapper = OceanUNet(base_model)
    wrapper.build(input_shape=(None, patch_size, patch_size, config['model']['in_channels']))
    wrapper.load_weights(checkpoint_path)
    model = base_model  # use the inner model for inference

    # ------------------------------------------------------------------
    # 3. Run inference — accumulate stats for RMSE and R²
    # ------------------------------------------------------------------
    print("Running inference on test set...")
    sum_sq_err = np.zeros(n_depths * 2)
    sum_y = np.zeros(n_depths * 2)
    sum_y2 = np.zeros(n_depths * 2)
    n_pixels = 0

    for i, t in enumerate(common_times):
        # Build input (H, W, 8)
        sst = ds_sst['analysed_sst'].sel(time=t).values
        sss = np.squeeze(ds_sss['sos'].sel(time=t).values)
        sla = ds_sla[sla_var].sel(time=t).values
        ugos = ds_sla['ugos'].sel(time=t).values
        vgos = ds_sla['vgos'].sel(time=t).values
        taux_var = 'surface_downward_eastward_stress' if 'surface_downward_eastward_stress' in ds_wind else 'eastward_stress'
        tauy_var = 'surface_downward_northward_stress' if 'surface_downward_northward_stress' in ds_wind else 'northward_stress'
        taux = ds_wind[taux_var].sel(time=t).values
        tauy = ds_wind[tauy_var].sel(time=t).values
        curl = ds_wind['wind_stress_curl'].sel(time=t).values
        x = np.nan_to_num(np.stack([sst, sss, sla, ugos, vgos, taux, tauy, curl], axis=-1), nan=0.0).astype(np.float32)

        # Build target (H, W, 52)
        gt = np.transpose(ds_glorys[temp_var].sel(time=t).values, (1, 2, 0))
        gs = np.transpose(ds_glorys[salt_var].sel(time=t).values, (1, 2, 0))
        y_true = np.nan_to_num(np.concatenate([gt, gs], axis=-1), nan=0.0).astype(np.float32)

        # Pad for UNet
        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0.0)
            y_true_padded = np.pad(y_true, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0.0)
        else:
            y_true_padded = y_true

        y_pred = model.predict(x[np.newaxis], verbose=0)[0]

        # Crop back to original spatial extent before computing metrics
        y_pred = y_pred[:lat_size, :lon_size, :]
        y_true = y_true[:lat_size, :lon_size, :]

        n = lat_size * lon_size
        sum_sq_err += np.sum((y_true - y_pred) ** 2, axis=(0, 1))
        sum_y += np.sum(y_true, axis=(0, 1))
        sum_y2 += np.sum(y_true ** 2, axis=(0, 1))
        n_pixels += n

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(common_times)} days...")

    # Compute final metrics
    mse = sum_sq_err / n_pixels
    rmse = np.sqrt(mse)
    ss_tot = sum_y2 - (sum_y ** 2) / n_pixels
    r2 = 1.0 - sum_sq_err / (ss_tot + 1e-8)

    rmse_T, rmse_S = rmse[:n_depths], rmse[n_depths:]
    r2_T, r2_S = r2[:n_depths], r2[n_depths:]

    print(f"\n  Mean Test RMSE — T: {rmse_T.mean():.4f}, S: {rmse_S.mean():.4f}")
    print(f"  Mean Test R²   — T: {r2_T.mean():.4f}, S: {r2_S.mean():.4f}")

    # ------------------------------------------------------------------
    # 4. Save UNet metrics into the shared JSON
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, 'metrics_bob_1_4.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

    metrics["UNet2D"] = {
        "RMSE_T": rmse_T.tolist(), "RMSE_S": rmse_S.tolist(),
        "R2_T": r2_T.tolist(), "R2_S": r2_S.tolist()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"  Metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # 5. Generate comparative Depth vs RMSE plots
    # ------------------------------------------------------------------
    print("Generating depth profile plots...")
    for target, r2_key, label, unit in [
        ('RMSE_T', 'R2_T', 'Temperature', '°C'),
        ('RMSE_S', 'R2_S', 'Salinity', 'psu')
    ]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'{label} — All Models (Test Set)', fontsize=14)

        # RMSE panel
        for name, met in metrics.items():
            if target in met:
                lw = 2.5 if name == 'UNet2D' else 1.5
                marker = 's' if name == 'UNet2D' else 'o'
                ax1.plot(met[target], depths, f'-{marker}', label=name, linewidth=lw)
        ax1.invert_yaxis()
        ax1.set_xlabel(f'RMSE (Z-Score Normalized {unit})')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title('RMSE vs Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # R² panel (only models that have it)
        for name, met in metrics.items():
            if r2_key in met:
                lw = 2.5 if name == 'UNet2D' else 1.5
                marker = 's' if name == 'UNet2D' else 'o'
                ax2.plot(met[r2_key], depths, f'-{marker}', label=name, linewidth=lw)
        ax2.invert_yaxis()
        ax2.set_xlabel('R²')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('R² vs Depth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        suffix = target[-1]
        out_path = os.path.join(results_dir, f'{model_name}_depth_profile_{suffix}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    ds_sst.close(); ds_sss.close(); ds_sla.close(); ds_wind.close(); ds_glorys.close()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained 2D-UNet")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
