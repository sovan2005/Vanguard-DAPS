import os
import sys
import requests
import numpy as np
import traceback
from PIL import Image, ImageDraw
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
ORIGINALS_DIR = BASE_DIR / "data" / "originals"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
MODEL_PATH = MODELS_DIR / "sscd_disc_mixup.torchscript.pt"

# remove partial
if MODEL_PATH.exists() and MODEL_PATH.stat().st_size < 80000000:
    print("Detected partial model file. Removing to restart download.")
    MODEL_PATH.unlink()

def download_file(url, path):
    if path.exists() and path.stat().st_size > 80000000:
         print(f"{path.name} already exists.")
         return
    print(f"Downloading {url} to {path.name}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, headers=headers, timeout=60)
        response.raise_for_status()
        with open(path, 'wb') as f:
             for chunk in response.iter_content(chunk_size=8192):
                  f.write(chunk)
        print(f"Downloaded to {path}")
    except Exception as e:
        print(f"FATAL: Failed to download model: {e}")
        if path.exists(): path.unlink()
        raise

def make_test_image(name, base_color):
    try:
        # Generates a gradient + noise image that is rich in features for SSCD
        arr = np.zeros((600, 800, 3), dtype=np.float32)
        
        # create some gradients
        x = np.linspace(0, 1, 800)
        y = np.linspace(0, 1, 600)
        xv, yv = np.meshgrid(x, y)
        
        arr[:, :, 0] = xv * base_color[0] + np.random.normal(0, 20, (600, 800))
        arr[:, :, 1] = yv * base_color[1] + np.random.normal(0, 20, (600, 800))
        arr[:, :, 2] = (xv + yv)/2 * base_color[2] + np.random.normal(0, 20, (600, 800))
        
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        
        # Draw some shapes to create strong features
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        d.ellipse([(200, 150), (400, 350)], fill=(base_color[2], base_color[0], base_color[1]))
        d.rectangle([(500, 300), (700, 500)], fill=(base_color[1], base_color[2], base_color[0]))
        
        save_path = ORIGINALS_DIR / name
        img.save(save_path)
        print(f"Generated base image: {name}")
    except Exception as e:
        print(f"ERROR: Failed to generate {name}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        download_file(MODEL_URL, MODEL_PATH)
        print("Generating synthetic test images...")
        np.random.seed(42)
        make_test_image("sports_01.jpg", (255, 100, 50))
        make_test_image("sports_02.jpg", (50, 150, 200))
        make_test_image("sports_03.jpg", (100, 200, 50))
        print("Setup complete.")
    except Exception as e:
        print(f"SETUP FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
