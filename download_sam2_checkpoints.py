import os
import urllib.request

# SAM 2.1 checkpoint base URL
BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# All checkpoints to download
CHECKPOINTS = {
    "sam2.1_hiera_tiny.pt": f"{BASE_URL}/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": f"{BASE_URL}/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": f"{BASE_URL}/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": f"{BASE_URL}/sam2.1_hiera_large.pt",
}

# Save directory
SAVE_DIR = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

def download(url, save_path):
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Saved to {save_path}\n")
    except Exception as e:
        print(f"❌ Failed to download {url}\nError: {e}")

def main():
    print("🚀 Starting SAM 2.1 checkpoint download...\n")
    for filename, url in CHECKPOINTS.items():
        save_path = os.path.join(SAVE_DIR, filename)
        if os.path.exists(save_path):
            print(f"⏩ Already exists: {save_path}")
            continue
        download(url, save_path)
    print("🎉 All checkpoints downloaded successfully!")

if __name__ == "__main__":
    main()
