import os
import json
import numpy as np
import xarray as xr
import warnings

def main():
    warnings.filterwarnings('ignore')
    print("=== CMEMS SLA / UGOS / VGOS Preprocessing (Bay of Bengal 1/4°) ===")
    
    with open(r'd:\Mlpr_project\target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = r'd:\Mlpr_project\data\bob_1_4_cache'
    os.makedirs(output_dir, exist_ok=True)
    
    sla_path = os.path.join(base_dir, 'SLA', 'cmems_sla_ugos_vgos_2023_2025.nc')
    output_path = os.path.join(output_dir, 'cmems_sla_ugos_vgos_preprocessed.nc')
    stats_path = os.path.join(output_dir, 'sla_uv_normalization_stats.json')
    
    if not os.path.exists(sla_path):
        print(f"Error: {sla_path} not found.")
        return
        
    print("Loading SLA raw dataset...")
    sla_ds = xr.open_dataset(sla_path)
    
    var_sla = 'sla' if 'sla' in sla_ds else 'adt'
    var_u = 'ugos' if 'ugos' in sla_ds else 'ugosa'
    var_v = 'vgos' if 'vgos' in sla_ds else 'vgosa'
    
    print("1. Defining Bay of Bengal 1/4° target grid...")
    target_lat = np.arange(5.0, 25.01, 0.25)
    target_lon = np.arange(80.0, 100.01, 0.25)
    
    print("2. Cropping raw data and regridding...")
    lat_dim = 'latitude' if 'latitude' in sla_ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in sla_ds.dims else 'lon'
    
    sla_ds = sla_ds.sel({lat_dim: slice(4.5, 25.5), lon_dim: slice(79.5, 100.5)})
    sla_interp = sla_ds.interp({lat_dim: target_lat, lon_dim: target_lon}, method='linear')
    if lat_dim == 'lat': sla_interp = sla_interp.rename({'lat': 'latitude'})
    if lon_dim == 'lon': sla_interp = sla_interp.rename({'lon': 'longitude'})
    
    print("3. Clipping Velocity Outliers...")
    sla_interp[var_u] = sla_interp[var_u].clip(-2.5, 2.5)
    sla_interp[var_v] = sla_interp[var_v].clip(-2.5, 2.5)
    
    print("4. Computing Train-Only Z-Score Normalization...")
    train_slice = sla_interp.sel(time=slice('2023-01-01', '2024-12-31'))
    
    stats = {}
    for var in [var_sla, var_u, var_v]:
        mean_val = float(train_slice[var].mean().values)
        std_val = float(train_slice[var].std().values)
        
        if var == var_sla:
            mean_val = 0.0 # SLA is already de-meaned
            
        stats[var] = {"mean": mean_val, "std": std_val}
        sla_interp[var] = (sla_interp[var] - mean_val) / std_val
        
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nSaving final preprocessed dataset to {output_path}...")
    sla_interp.to_netcdf(output_path)
    print("SUCCESS! SLA/UGOS/VGOS Preprocessing Complete.")

if __name__ == "__main__":
    main()
