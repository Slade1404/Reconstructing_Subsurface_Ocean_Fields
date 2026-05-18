import os
import xarray as xr
import copernicusmarine

def main():
    print("=== CMEMS SLA / UGOS / VGOS Downloader ===")
    
    with open('target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = os.path.join(base_dir, 'SLA')
    os.makedirs(output_dir, exist_ok=True)
    
    # Bounding box matching GLORYS
    lon_min, lon_max = 60.0, 100.0
    lat_min, lat_max = 2.0, 25.0
    
    # Common subset parameters
    variables = ["sla", "ugos", "vgos"]
    
    # ---- Part 1: Multi-Year archive (covers up to ~Oct 2025) ----
    my_path = os.path.join(output_dir, 'sla_multiyear.nc')
    if not os.path.exists(my_path):
        print("\n[1/3] Downloading Multi-Year archive (May 2025 → Oct 2025)...")
        copernicusmarine.subset(
            dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            variables=variables,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            start_datetime="2023-01-01 00:00:00",
            end_datetime="2025-12-31 23:59:59",
            output_directory=output_dir,
            output_filename="sla_multiyear.nc"
        )
    else:
        print("[1/3] Multi-Year file already exists, skipping.")
    
    # ---- Part 2: Near-Real-Time archive (covers Oct 2025 → present) ----
    nrt_path = os.path.join(output_dir, 'sla_nrt.nc')
    if not os.path.exists(nrt_path):
        print("\n[2/3] Downloading Near-Real-Time archive (Oct 2025 → Mar 2026)...")
        copernicusmarine.subset(
            dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
            variables=variables,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            start_datetime="2025-10-19 00:00:00",
            end_datetime="2025-12-31 23:59:59",
            output_directory=output_dir,
            output_filename="sla_nrt.nc"
        )
    else:
        print("[2/3] NRT file already exists, skipping.")
    
    # ---- Part 3: Merge both into one seamless file ----
    print("\n[3/3] Merging Multi-Year + NRT into single continuous dataset...")
    ds_my = xr.open_dataset(my_path)
    ds_nrt = xr.open_dataset(nrt_path)
    
    merged = xr.concat([ds_my, ds_nrt], dim='time')
    merged = merged.sortby('time')
    # Drop any duplicate timesteps from overlap
    _, unique_idx = np.unique(merged.time.values, return_index=True)
    merged = merged.isel(time=unique_idx)
    
    final_path = os.path.join(output_dir, 'cmems_sla_ugos_vgos_2023_2025.nc')
    merged.to_netcdf(final_path)
    
    print(f"\nSUCCESS! Merged dataset saved to: {final_path}")
    print(f"  Time range: {merged.time.values[0]} to {merged.time.values[-1]}")
    print(f"  Total days: {len(merged.time)}")
    print(f"  Variables: {list(merged.data_vars)}")
    
    ds_my.close()
    ds_nrt.close()
    merged.close()

if __name__ == "__main__":
    import numpy as np
    main()
