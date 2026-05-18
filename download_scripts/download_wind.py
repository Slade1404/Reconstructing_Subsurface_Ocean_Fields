import os
import copernicusmarine

def main():
    print("=== CMEMS WIND L4 NRT Downloader ===")
    
    with open(r'd:\Mlpr_project\target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    output_dir = os.path.join(base_dir, 'WIND')
    os.makedirs(output_dir, exist_ok=True)
    
    lon_min, lon_max = 60.0, 100.0
    lat_min, lat_max = 2.0, 25.0
    
    print("Downloading WIND_GLO_PHY_L4_MY (hourly)...")
    print("Time: 2023-01-01 to 2025-12-31")
    
    # We request surface_downward_eastward_stress and surface_downward_northward_stress
    # The MY dataset does not include wind_stress_curl, so we compute it manually in preprocessing
    copernicusmarine.subset(
        dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        variables=["surface_downward_eastward_stress", "surface_downward_northward_stress"],
        minimum_longitude=lon_min, maximum_longitude=lon_max,
        minimum_latitude=lat_min, maximum_latitude=lat_max,
        start_datetime="2023-01-01 00:00:00",
        end_datetime="2025-12-31 23:59:59",
        output_directory=output_dir,
        output_filename="cmems_wind_2023_2025.nc",
    )
    print(f"\nSUCCESS! Wind saved to: {os.path.join(output_dir, 'cmems_wind_2023_2025.nc')}")

if __name__ == "__main__":
    main()
