"""
Creates a small 16x16 spatial crop of the BoB 1/4° data for quick pipeline testing.
Original data is NOT modified.
"""
import xarray as xr
import os

def main():
    src_dir = r'd:\Mlpr_project\data\bob_1_4_cache'
    dst_dir = r'd:\Mlpr_project\data\bob_1_4_test_crop'
    os.makedirs(dst_dir, exist_ok=True)

    crop = 16  # 16x16 spatial crop

    files = [
        'cmems_sst_preprocessed.nc',
        'cmems_sss_preprocessed.nc',
        'cmems_sla_ugos_vgos_preprocessed.nc',
        'cmems_glorys_preprocessed.nc'
    ]

    for f in files:
        print(f"Cropping {f} to {crop}x{crop}...")
        ds = xr.open_dataset(os.path.join(src_dir, f))
        ds_crop = ds.isel(latitude=slice(0, crop), longitude=slice(0, crop))
        ds_crop.to_netcdf(os.path.join(dst_dir, f))
        ds.close()
        ds_crop.close()

    print(f"\nDone! Cropped files saved to {dst_dir}")

if __name__ == "__main__":
    main()
