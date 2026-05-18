import os
import copernicusmarine

def main():
    print("=== CMEMS GLORYS 3D (NRT) Downloader ===")
    
    with open('target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = os.path.join(base_dir, 'GLORYS')
    os.makedirs(output_dir, exist_ok=True)
    
    # Target Bay of Bengal 1/4° grid bounds (plus a small buffer)
    lon_min, lon_max = 79.5, 100.5
    lat_min, lat_max = 4.5, 25.5
    depth_min, depth_max = 0.0, 200.0
    
    print("Downloading GLORYS 3D Temperature & Salinity (NRT, 1/12°, daily)...")
    print("Time: 2023-01-01 to 2025-12-31")
    print("Depth: 0m to 200m")
    
    try:
        print("\nDownloading Temperature (thetao)...")
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            minimum_depth=depth_min, maximum_depth=depth_max,
            start_datetime="2023-01-01 00:00:00",
            end_datetime="2025-12-31 23:59:59",
            output_directory=output_dir,
            output_filename="cmems_glorys_bob_thetao_2023_2025.nc"
        )
        
        print("\nDownloading Salinity (so)...")
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=["so"],
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            minimum_depth=depth_min, maximum_depth=depth_max,
            start_datetime="2023-01-01 00:00:00",
            end_datetime="2025-12-31 23:59:59",
            output_directory=output_dir,
            output_filename="cmems_glorys_bob_so_2023_2025.nc"
        )
        
        print(f"\nSUCCESS! GLORYS saved to: {output_dir}")
    except Exception as e:
        print(f"Error downloading: {e}")

if __name__ == "__main__":
    main()
