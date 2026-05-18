import os
import json
import numpy as np
import xarray as xr
import warnings

def main():
    warnings.filterwarnings('ignore')
    print("=== CMEMS GLORYS 3D Preprocessing (Bay of Bengal 1/4°) ===")
    
    with open(r'd:\Mlpr_project\target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    raw_dir = os.path.join(base_dir, 'GLORYS')
    output_dir = r'd:\Mlpr_project\data\bob_1_4_cache'
    os.makedirs(output_dir, exist_ok=True)
    
    thetao_path = os.path.join(raw_dir, 'cmems_glorys_bob_thetao_2023_2025.nc')
    so_path = os.path.join(raw_dir, 'cmems_glorys_bob_so_2023_2025.nc')
    output_path = os.path.join(output_dir, 'cmems_glorys_preprocessed.nc')
    stats_path = os.path.join(output_dir, 'glorys_normalization_stats.json')
    
    if not os.path.exists(thetao_path) or not os.path.exists(so_path):
        print("Error: Raw GLORYS files not found in raw_cache.")
        return
        
    print("1. Loading raw GLORYS datasets...")
    # Load into memory since we already downloaded the subsetted version (~4GB total)
    ds_thetao = xr.open_dataset(thetao_path)
    ds_so = xr.open_dataset(so_path)
    
    # Merge them into a single dataset
    ds = xr.merge([ds_thetao, ds_so])
    
    print("2. Defining Bay of Bengal 1/4° target grid...")
    target_lat = np.arange(5.0, 25.01, 0.25)
    target_lon = np.arange(80.0, 100.01, 0.25)
    
    print("3. Interpolating to 1/4° grid...")
    # No extrapolation needed because our local download went up to 100.5E!
    ds_interp = ds.interp(
        latitude=target_lat,
        longitude=target_lon,
        method='linear'
    )
    
    # Ensure time is sorted
    ds_interp = ds_interp.sortby('time')
    
    print("4. Computing Train-Only Z-Score Normalization (Depth-wise)...")
    # Training split: 2023-01-01 to 2024-12-31
    train_ds = ds_interp.sel(time=slice('2023-01-01', '2024-12-31'))
    
    # Compute mean and std per depth level (dims: time, lat, lon)
    # This leaves an array of shape (26,) for both mean and std
    mean_thetao = train_ds['thetao'].mean(dim=['time', 'latitude', 'longitude']).values
    std_thetao = train_ds['thetao'].std(dim=['time', 'latitude', 'longitude']).values
    
    mean_so = train_ds['so'].mean(dim=['time', 'latitude', 'longitude']).values
    std_so = train_ds['so'].std(dim=['time', 'latitude', 'longitude']).values
    
    # Replace any 0 standard deviations with 1 to prevent division by zero
    std_thetao[std_thetao == 0] = 1.0
    std_so[std_so == 0] = 1.0
    
    # Apply normalization using xarray broadcasting
    # We must construct xarray DataArrays for the mean/std to broadcast along the 'depth' dimension
    mean_thetao_da = xr.DataArray(mean_thetao, coords=[ds_interp.depth], dims=['depth'])
    std_thetao_da = xr.DataArray(std_thetao, coords=[ds_interp.depth], dims=['depth'])
    
    mean_so_da = xr.DataArray(mean_so, coords=[ds_interp.depth], dims=['depth'])
    std_so_da = xr.DataArray(std_so, coords=[ds_interp.depth], dims=['depth'])
    
    ds_interp['thetao_norm'] = (ds_interp['thetao'] - mean_thetao_da) / std_thetao_da
    ds_interp['so_norm'] = (ds_interp['so'] - mean_so_da) / std_so_da
    
    # Keep only the normalized variables
    ds_final = ds_interp[['thetao_norm', 'so_norm']]
    
    print(f"   Saving final dataset to {output_path}...")
    ds_final.to_netcdf(output_path)
    
    # Save the depth-wise stats (convert to lists for JSON)
    stats = {
        'thetao': {
            'mean': mean_thetao.tolist(),
            'std': std_thetao.tolist(),
            'depths': ds_interp.depth.values.tolist()
        },
        'so': {
            'mean': mean_so.tolist(),
            'std': std_so.tolist(),
            'depths': ds_interp.depth.values.tolist()
        }
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("SUCCESS! GLORYS Preprocessing Complete.")
    print(f"Final shape: time={len(ds_final.time)}, depth={len(ds_final.depth)}, "
          f"lat={len(ds_final.latitude)}, lon={len(ds_final.longitude)}")

if __name__ == "__main__":
    main()
