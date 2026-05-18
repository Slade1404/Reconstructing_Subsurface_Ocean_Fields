"""
Dataset pipeline for the 2D-UNet ocean reconstruction model.

Provides a TensorFlow data generator that loads preprocessed NetCDF files
(SST, SSS, SLA, UGOS, VGOS) as input features and GLORYS 3D fields
(temperature + salinity at multiple depth levels) as targets.

Supports two spatial modes:
  - Patch mode: randomly crops NxN patches from a larger map (for global data)
  - Full-map mode: loads the entire spatial domain and zero-pads to patch_size
    for UNet compatibility (for regional data like Bay of Bengal)

Argo float mask loading is conditional and controlled via config.
"""
import tensorflow as tf
import numpy as np
import xarray as xr


class OceanDatasetGenerator:
    """
    Lazily yields (input, target) pairs from preprocessed NetCDF files.

    Input:  (H, W, 8) — SST, SSS, SLA, UGOS, VGOS, TAUX, TAUY, CURL
    Target: (H, W, out_channels) — [thetao at D depths, so at D depths]
    Mask:   (H, W, 1) — Argo float coverage (all zeros if Argo is disabled)

    NaN values (land pixels) are replaced with 0.0.
    """

    def __init__(self, config_data, is_train=True):
        """
        Args:
            config_data: The 'data' section of the YAML config.
            is_train: If True, uses time_train range; otherwise time_test.
        """
        self.patch_size = config_data.get('patch_size', 64)
        self.full_map = config_data.get('full_map', False)

        # Select time range
        time_range = config_data['time_train'] if is_train else config_data['time_test']
        self.start_date, self.end_date = time_range
        split = 'Train' if is_train else 'Test'

        print(f"[{split}] Loading datasets from {self.start_date} to {self.end_date}...")

        # Load input feature datasets
        self.ds_sst = xr.open_dataset(config_data['sst_path']).sel(time=slice(self.start_date, self.end_date))
        self.ds_sss = xr.open_dataset(config_data['sss_path']).sel(time=slice(self.start_date, self.end_date))
        self.ds_sla_uv = xr.open_dataset(config_data['sla_uv_path']).sel(time=slice(self.start_date, self.end_date))
        self.ds_wind = xr.open_dataset(config_data['wind_path']).sel(time=slice(self.start_date, self.end_date))

        # Load target dataset (GLORYS reanalysis)
        self.ds_glorys = xr.open_dataset(config_data['glorys_path']).sel(time=slice(self.start_date, self.end_date))

        # Conditional Argo loading (only if a valid path is provided)
        self.use_argo = bool(config_data.get('argo_path', ''))
        if self.use_argo:
            try:
                self.ds_argo = xr.open_dataset(config_data['argo_path']).sel(time=slice(self.start_date, self.end_date))
                if len(self.ds_argo.time) == 0:
                    self.use_argo = False
            except KeyError:
                # If slice fails completely
                self.use_argo = False

        # Auto-detect variable names for backward compatibility
        self.glorys_temp_var = 'thetao_norm' if 'thetao_norm' in self.ds_glorys else 'thetao'
        self.glorys_salt_var = 'so_norm' if 'so_norm' in self.ds_glorys else 'so'
        self.sla_var = 'sla' if 'sla' in self.ds_sla_uv else 'adt'
        self.taux_var = 'surface_downward_eastward_stress' if 'surface_downward_eastward_stress' in self.ds_wind else 'eastward_stress'
        self.tauy_var = 'surface_downward_northward_stress' if 'surface_downward_northward_stress' in self.ds_wind else 'northward_stress'

        # Find overlapping timesteps across all datasets
        time_sets = [set(ds.time.values) for ds in
                     [self.ds_sst, self.ds_sss, self.ds_sla_uv, self.ds_wind, self.ds_glorys]]
        if self.use_argo:
            time_sets.append(set(self.ds_argo.time.values))
        self.common_times = sorted(set.intersection(*time_sets))

        if not self.common_times:
            raise ValueError(f"No overlapping times found in range {time_range}")

        print(f"[{split}] {len(self.common_times)} overlapping days found.")

        # Spatial dimensions of the unpadded data
        self.lat_size = self.ds_sst.sizes['latitude']
        self.lon_size = self.ds_sst.sizes['longitude']

        # Number of samples to yield per epoch
        default = 200000 if is_train else 10000
        self.num_samples = config_data.get(
            'train_samples' if is_train else 'val_samples', default)

    def __call__(self):
        """Generator yielding (x, {"dense_target": y, "argo_mask": mask})."""
        for _ in range(self.num_samples):
            # 1. Randomly sample a timestep
            t = np.random.choice(self.common_times)

            # 2. Spatial selection
            if self.full_map:
                lat_sl = slice(0, self.lat_size)
                lon_sl = slice(0, self.lon_size)
            else:
                i = np.random.randint(0, self.lat_size - self.patch_size + 1)
                j = np.random.randint(0, self.lon_size - self.patch_size + 1)
                lat_sl = slice(i, i + self.patch_size)
                lon_sl = slice(j, j + self.patch_size)

            # 3. Extract surface features → (H, W, 8)
            sst = self.ds_sst['analysed_sst'].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            sss = np.squeeze(self.ds_sss['sos'].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values)
            sla = self.ds_sla_uv[self.sla_var].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            ugos = self.ds_sla_uv['ugos'].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            vgos = self.ds_sla_uv['vgos'].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            taux = self.ds_wind[self.taux_var].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            tauy = self.ds_wind[self.tauy_var].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            curl = self.ds_wind['wind_stress_curl'].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values

            x = np.nan_to_num(np.stack([sst, sss, sla, ugos, vgos, taux, tauy, curl], axis=-1), nan=0.0)

            # 4. Extract GLORYS targets → (H, W, 2*D)
            #    Raw shape is (depth, lat, lon); transpose to (lat, lon, depth)
            gt = self.ds_glorys[self.glorys_temp_var].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            gs = self.ds_glorys[self.glorys_salt_var].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
            if gt.ndim == 3:
                gt = np.transpose(gt, (1, 2, 0))
                gs = np.transpose(gs, (1, 2, 0))
            y = np.nan_to_num(np.concatenate([gt, gs], axis=-1), nan=0.0)

            # 5. Argo mask and target → (H, W, 1) and (H, W, D)
            if self.use_argo:
                # Auto-detect variable name (TEMP vs temp)
                argo_temp_var = 'TEMP' if 'TEMP' in self.ds_argo else 'temp'
                argo = self.ds_argo[argo_temp_var].sel(time=t).isel(latitude=lat_sl, longitude=lon_sl).values
                if argo.ndim == 3:
                    argo = np.transpose(argo, (1, 2, 0))
                mask = (~np.isnan(argo)).any(axis=-1, keepdims=True).astype(np.float32)
                argo_target = np.nan_to_num(argo, nan=0.0)
            else:
                mask = np.zeros((x.shape[0], x.shape[1], 1), dtype=np.float32)
                argo_target = np.zeros((x.shape[0], x.shape[1], 26), dtype=np.float32)

            # 6. Pad to patch_size if using full-map mode
            if self.full_map:
                ph = self.patch_size - x.shape[0]
                pw = self.patch_size - x.shape[1]
                if ph > 0 or pw > 0:
                    x = np.pad(x, ((0, ph), (0, pw), (0, 0)), constant_values=0.0)
                    y = np.pad(y, ((0, ph), (0, pw), (0, 0)), constant_values=0.0)
                    mask = np.pad(mask, ((0, ph), (0, pw), (0, 0)), constant_values=0.0)
                    argo_target = np.pad(argo_target, ((0, ph), (0, pw), (0, 0)), constant_values=0.0)

            yield x.astype(np.float32), {
                "dense_target": y.astype(np.float32),
                "argo_mask": mask,
                "argo_target": argo_target.astype(np.float32)
            }


def create_dataset(config_data, out_channels, is_train=True):
    """
    Creates a batched, prefetched tf.data.Dataset from the ocean data generator.

    Args:
        config_data: The 'data' section of the YAML config.
        out_channels: Number of output channels (e.g. 52 = 26 temp + 26 salt).
        is_train: If True, loads training split; otherwise test/validation.

    Returns:
        A tf.data.Dataset yielding (x, {"dense_target": y, "argo_mask": mask}).
    """
    patch_size = config_data.get('patch_size', 64)
    batch_size = config_data.get('batch_size', 4)
    in_channels = config_data.get('in_channels', 5)

    generator = OceanDatasetGenerator(config_data, is_train)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(patch_size, patch_size, in_channels), dtype=tf.float32),
            {
                "dense_target": tf.TensorSpec(shape=(patch_size, patch_size, out_channels), dtype=tf.float32),
                "argo_mask": tf.TensorSpec(shape=(patch_size, patch_size, 1), dtype=tf.float32),
                "argo_target": tf.TensorSpec(shape=(patch_size, patch_size, 26), dtype=tf.float32)
            }
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
