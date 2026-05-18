import os
import xarray as xr

def main():
    print("=== ETOPO Bathymetry Downloader ===")
    
    output_dir = r'd:\Mlpr_project\data\raw_cache'
    os.makedirs(output_dir, exist_ok=True)
    
    lon_min, lon_max = 60.0, 100.0
    lat_min, lat_max = 2.0, 25.0
    
    print("Connecting to PACIOOS ETOPO5 OPeNDAP server...")
    url = "https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/etopo5"
    
    try:
        ds = xr.open_dataset(url)
        print("Dataset loaded. Available coordinates:")
        print(list(ds.coords))
        
        # Determine coordinate names (often ETOPO05_X and ETOPO05_Y, or lon/lat)
        x_coord = 'ETOPO05_X' if 'ETOPO05_X' in ds.coords else 'lon'
        y_coord = 'ETOPO05_Y' if 'ETOPO05_Y' in ds.coords else 'lat'
        
        print(f"Subsetting bounding box: Lon [{lon_min}, {lon_max}], Lat [{lat_min}, {lat_max}]")
        
        # Subsetting using .sel with slices
        ds_subset = ds.sel({
            x_coord: slice(lon_min, lon_max),
            y_coord: slice(lat_min, lat_max)
        })
        
        # Save to disk
        out_path = os.path.join(output_dir, 'etopo5_bathymetry.nc')
        ds_subset.to_netcdf(out_path)
        print(f"\nSUCCESS! Bathymetry saved to: {out_path}")
        
    except Exception as e:
        print(f"Failed to download bathymetry via OPeNDAP: {e}")
        print("Please ensure internet connection is stable or PACIOOS server is online.")

if __name__ == "__main__":
    main()
