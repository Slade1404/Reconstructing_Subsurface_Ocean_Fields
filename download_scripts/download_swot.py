import os
import subprocess

def main():
    print("=== SWOT Regional Data Downloader (PO.DAAC Subscriber) ===")
    
    # Target directory on Google Drive
    with open('target.txt', 'r', encoding='utf-16') as f:
        base_dir = f.read().strip()
        
    target_dir = os.path.join(base_dir, 'SWOT')
    os.makedirs(target_dir, exist_ok=True)
    
    # Bounding Box for Arabian Sea / Bay of Bengal
    # Format: W Longitude,S Latitude,E Longitude,N Latitude
    bbox = "60,2,100,25"
    start_date = "2026-03-24T00:00:00Z"
    end_date = "2026-03-25T00:00:00Z"
    collection = "SWOT_L2_LR_SSH_BASIC_D"
    
    print(f"Downloading dataset: {collection}")
    print(f"Region Bounds: {bbox}")
    print(f"Time Range: {start_date} to {end_date}")
    print(f"Target Directory: {target_dir}")
    
    # Construct the podaac-data-downloader command
    # Use -f to force download in case of prior interrupted attempts
    cmd = [
        "podaac-data-downloader",
        "-c", collection,
        "-d", target_dir,
        "-sd", start_date,
        "-ed", end_date,
        f"-b={bbox}"
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("This may take some time depending on data volume...\n")
    
    try:
        # Run the command and stream output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        
        if process.returncode == 0:
            print("\nDownload completed successfully!")
        else:
            print(f"\nDownloader failed with exit code {process.returncode}")
            
    except FileNotFoundError:
        print("\nError: 'podaac-data-downloader' command not found. Is podaac-data-subscriber installed?")
    except Exception as e:
        print(f"\nError executing downloader: {e}")

if __name__ == "__main__":
    main()
