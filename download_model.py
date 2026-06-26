#!/usr/bin/env python3
"""
Download latest model from GitHub Releases to VPS.
Run this after training to deploy new model.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path

REPO = "antonoved2/rl_polymarket"
MODEL_DIR = Path("/opt/rl_trader/models")

def download_latest_model():
    """Download latest model from GitHub Releases."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get latest release
    url = f"https://api.github.com/repos/{REPO}/releases/latest"
    print(f"Fetching latest release info...")
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code}")
        return None
    
    release = resp.json()
    tag = release["tag_name"]
    print(f"Latest release: {tag}")
    
    # Find model asset
    for asset in release.get("assets", []):
        name = asset["name"]
        if name.endswith(".zip") and "ppo" in name.lower():
            download_url = asset["browser_download_url"]
            dest = MODEL_DIR / name
            
            print(f"Downloading {name}...")
            resp = requests.get(download_url, timeout=60)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                print(f"Saved to {dest} ({len(resp.content) / 1024 / 1024:.1f} MB)")
                return str(dest)
            else:
                print(f"Download failed: {resp.status_code}")
    
    print("No model found in release")
    return None


def download_from_repo(model_name):
    """Download model directly from repo (if stored in models/)."""
    url = f"https://raw.githubusercontent.com/{REPO}/master/models/{model_name}"
    dest = MODEL_DIR / model_name
    
    print(f"Downloading {model_name} from repo...")
    resp = requests.get(url, timeout=60)
    if resp.status_code == 200:
        dest.write_bytes(resp.content)
        print(f"Saved to {dest} ({len(resp.content) / 1024 / 1024:.1f} MB)")
        return str(dest)
    else:
        print(f"Download failed: {resp.status_code}")
        return None


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    
    if model:
        download_from_repo(model)
    else:
        result = download_latest_model()
        if result:
            print(f"\nModel ready: {result}")
        else:
            print("\nFailed to download model")
