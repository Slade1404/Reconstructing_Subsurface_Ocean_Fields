import os
import json
import numpy as np
import xarray as xr
import warnings

def main():
    warnings.filterwarnings('ignore')
    print("=== CMEMS SSS Preprocessing (Bay of Bengal 1/4°) ===")
    
    with open(r'd:\Mlpr_project\target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = r'd:\Mlpr_project\data\bob_1_4_cache'
    os.makedirs(output_dir, exist_ok=True)
    
    sss_path = os.path.join(base_dir, 'SSS', 'cmems_sss_2023_2025.nc')
    output_path = os.path.join(output_dir, 'cmems_sss_preprocessed.nc')
    stats_path = os.path.join(output_dir, 'sss_normalization_stats.json')
    
    if not os.path.exists(sss_path):
        print(f"Error: {sss_path} not found.")
        return
        
    print("Loading SSS raw dataset...")
    sss_ds = xr.open_dataset(sss_path)
    var_sss = 'sos' if 'sos' in sss_ds else 'sss'
    
    print(f"1. Clipping {var_sss} to [30, 40] psu...")
    sss_ds[var_sss] = sss_ds[var_sss].clip(30, 40)
    
    print("2. Defining Bay of Bengal 1/4° target grid...")
    target_lat = np.arange(5.0, 25.01, 0.25)
    target_lon = np.arange(80.0, 100.01, 0.25)
    
    print("3. Cropping raw data and regridding...")
    lat_dim = 'latitude' if 'latitude' in sss_ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in sss_ds.dims else 'lon'
    
    sss_ds = sss_ds.sel({lat_dim: slice(4.5, 25.5), lon_dim: slice(79.5, 100.5)})
    sss_ds = sss_ds.interp({lat_dim: target_lat, lon_dim: target_lon}, method='linear')
    if lat_dim == 'lat': sss_ds = sss_ds.rename({'lat': 'latitude'})
    if lon_dim == 'lon': sss_ds = sss_ds.rename({'lon': 'longitude'})
    
    print("4. Computing Train-Only Z-Score Normalization...")
    train_slice = sss_ds.sel(time=slice('2023-01-01', '2024-12-31'))
    mean_val = float(train_slice[var_sss].mean().values)
    std_val = float(train_slice[var_sss].std().values)
    
    stats = {var_sss: {"mean": mean_val, "std": std_val}}
    sss_ds[var_sss] = (sss_ds[var_sss] - mean_val) / std_val
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nSaving final preprocessed dataset to {output_path}...")
    sss_ds.to_netcdf(output_path)
    print("SUCCESS! SSS Preprocessing Complete.")

if __name__ == "__main__":
    main()
