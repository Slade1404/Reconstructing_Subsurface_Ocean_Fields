import os
import json
import numpy as np
import xarray as xr
import warnings

def main():
    warnings.filterwarnings('ignore')
    print("=== CMEMS SST Preprocessing (Bay of Bengal 1/4°) ===")
    
    with open(r'd:\Mlpr_project\target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = r'd:\Mlpr_project\data\bob_1_4_cache'
    os.makedirs(output_dir, exist_ok=True)
    
    sst_path = os.path.join(base_dir, 'SST', 'cmems_sst_2023_2025.nc')
    output_path = os.path.join(output_dir, 'cmems_sst_preprocessed.nc')
    stats_path = os.path.join(output_dir, 'sst_normalization_stats.json')
    
    if not os.path.exists(sst_path):
        print(f"Error: {sst_path} not found. Please download the raw data first.")
        return
        
    print("Loading SST raw dataset...")
    sst_ds = xr.open_dataset(sst_path)
    
    var_sst = 'analysed_sst'
    if float(sst_ds[var_sst].mean()) > 200:
        print("1. Converting SST from Kelvin to Celsius...")
        sst_ds[var_sst] = sst_ds[var_sst] - 273.15
        
    print("2. Clipping SST to [-2, 35] °C...")
    sst_ds[var_sst] = sst_ds[var_sst].clip(-2, 35)
    
    print("3. Defining Bay of Bengal 1/4° target grid...")
    target_lat = np.arange(5.0, 25.01, 0.25)
    target_lon = np.arange(80.0, 100.01, 0.25)
    
    print("4. Cropping raw data to bounding box and regridding...")
    lat_dim = 'latitude' if 'latitude' in sst_ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in sst_ds.dims else 'lon'
    
    # Pre-clip to save memory (with 0.5 deg buffer)
    sst_ds = sst_ds.sel({lat_dim: slice(4.5, 25.5), lon_dim: slice(79.5, 100.5)})
    
    sst_ds = sst_ds.interp({lat_dim: target_lat, lon_dim: target_lon}, method='linear')
    # Rename lat/lon if they were named differently so it's standardized
    if lat_dim == 'lat': sst_ds = sst_ds.rename({'lat': 'latitude'})
    if lon_dim == 'lon': sst_ds = sst_ds.rename({'lon': 'longitude'})
    
    print("5. Computing Train-Only Z-Score Normalization...")
    train_slice = sst_ds.sel(time=slice('2023-01-01', '2024-12-31'))
    mean_val = float(train_slice[var_sst].mean().values)
    std_val = float(train_slice[var_sst].std().values)
    
    stats = {var_sst: {"mean": mean_val, "std": std_val}}
    sst_ds[var_sst] = (sst_ds[var_sst] - mean_val) / std_val
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nSaving final preprocessed dataset to {output_path}...")
    sst_ds.to_netcdf(output_path)
    print("SUCCESS! SST Preprocessing Complete.")

if __name__ == "__main__":
    main()
