import os
import copernicusmarine

def main():
    print("=== CMEMS SSS Downloader ===")
    
    with open('target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = os.path.join(base_dir, 'SSS')
    os.makedirs(output_dir, exist_ok=True)
    
    # Bounding box matching GLORYS
    lon_min, lon_max = 60.0, 100.0
    lat_min, lat_max = 2.0, 25.0
    
    print("Downloading SSS NRT (Multi-Obs, 1/4°, daily)...")
    print("Time: 2023-01-01 to 2025-12-31")
    
    copernicusmarine.subset(
        dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
        variables=["sos"],
        minimum_longitude=lon_min, maximum_longitude=lon_max,
        minimum_latitude=lat_min, maximum_latitude=lat_max,
        start_datetime="2023-01-01 00:00:00",
        end_datetime="2025-12-31 23:59:59",
        output_directory=output_dir,
        output_filename="cmems_sss_2023_2025.nc"
    )
    
    print(f"\nSUCCESS! SSS saved to: {os.path.join(output_dir, 'cmems_sss_2023_2025.nc')}")

if __name__ == "__main__":
    main()
