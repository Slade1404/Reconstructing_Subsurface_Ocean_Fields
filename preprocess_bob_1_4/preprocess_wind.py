import os
import json
import numpy as np
import xarray as xr
import warnings

def compute_curl(tau_x, tau_y, lon, lat):
    lon2d, lat2d = np.meshgrid(lon, lat)
    R = 6371e3
    dx = np.gradient(lon2d, axis=1) * np.cos(np.deg2rad(lat2d)) * (np.pi/180) * R
    dy = np.gradient(lat2d, axis=0) * (np.pi/180) * R
    
    dtauy_dx = np.gradient(tau_y.values, axis=2) / dx[None, :, :]
    dtaux_dy = np.gradient(tau_x.values, axis=1) / dy[None, :, :]
    
    curl = dtauy_dx - dtaux_dy
    return curl

def main():
    warnings.filterwarnings('ignore')
    print("=== CMEMS WIND Preprocessing (Bay of Bengal 1/4°) ===")
    
    with open(r'd:\Mlpr_project\target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = r'd:\Mlpr_project\data\bob_1_4_cache'
    os.makedirs(output_dir, exist_ok=True)
    
    wind_path = os.path.join(base_dir, 'WIND', 'cmems_wind_2023_2025.nc')
    output_path = os.path.join(output_dir, 'cmems_wind_preprocessed.nc')
    stats_path = os.path.join(output_dir, 'wind_normalization_stats.json')
    
    if not os.path.exists(wind_path):
        print(f"Error: {wind_path} not found.")
        return
        
    print("1. Loading WIND raw dataset (Hourly)...")
    wind_ds = xr.open_dataset(wind_path)
    
    print("2. Resampling to Daily Means...")
    wind_daily = wind_ds.resample(time='1D').mean()
    
    lat_dim = 'latitude' if 'latitude' in wind_daily.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in wind_daily.dims else 'lon'
    
    # Pre-crop
    wind_daily = wind_daily.sel({lat_dim: slice(4.5, 25.5), lon_dim: slice(79.5, 100.5)})
    
    if 'wind_stress_curl' in wind_daily.data_vars:
        print("3. Native wind_stress_curl found!")
        curl = wind_daily['wind_stress_curl'].values
    else:
        print("3. Native curl missing. Computing from tau_x and tau_y...")
        tau_x = wind_daily['surface_downward_eastward_stress'] if 'surface_downward_eastward_stress' in wind_daily else wind_daily['eastward_stress']
        tau_y = wind_daily['surface_downward_northward_stress'] if 'surface_downward_northward_stress' in wind_daily else wind_daily['northward_stress']
        curl = compute_curl(tau_x, tau_y, wind_daily[lon_dim].values, wind_daily[lat_dim].values)
        wind_daily['wind_stress_curl'] = (['time', lat_dim, lon_dim], curl)
        
    print("4. Defining Bay of Bengal 1/4° target grid...")
    target_lat = np.arange(5.0, 25.01, 0.25)
    target_lon = np.arange(80.0, 100.01, 0.25)
    
    print("5. Regridding Wind/Curl to 1/4° grid...")
    wind_interp = wind_daily.interp({lat_dim: target_lat, lon_dim: target_lon}, method='linear')
    if lat_dim == 'lat': wind_interp = wind_interp.rename({'lat': 'latitude'})
    if lon_dim == 'lon': wind_interp = wind_interp.rename({'lon': 'longitude'})
    
    print("6. Computing Train-Only Z-Score Normalization...")
    train_slice = wind_interp.sel(time=slice('2023-01-01', '2024-12-31'))
    
    var_tau_x = 'surface_downward_eastward_stress' if 'surface_downward_eastward_stress' in wind_interp else 'eastward_stress'
    var_tau_y = 'surface_downward_northward_stress' if 'surface_downward_northward_stress' in wind_interp else 'northward_stress'
    vars_to_norm = [var_tau_x, var_tau_y, 'wind_stress_curl']
    
    stats = {}
    for var in vars_to_norm:
        if var not in wind_interp: continue
        mean_val = float(train_slice[var].mean().values)
        std_val = float(train_slice[var].std().values)
        if std_val == 0: std_val = 1e-8
        
        stats[var] = {"mean": mean_val, "std": std_val}
        wind_interp[var] = (wind_interp[var] - mean_val) / std_val
        
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nSaving final preprocessed dataset to {output_path}...")
    wind_interp.to_netcdf(output_path)
    print("SUCCESS! Wind Preprocessing Complete.")

if __name__ == "__main__":
    main()
