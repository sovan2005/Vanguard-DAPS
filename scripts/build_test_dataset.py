import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import uuid
import hashlib

# Fix relative imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.embedder import embedder
from core.indexer import faiss_index
from db.models import Asset
from db.database import get_session, init_db

ORIGINALS_DIR = BASE_DIR / "data" / "originals"
QUERIES_DIR   = BASE_DIR / "data" / "queries"
QUERIES_DIR.mkdir(parents=True, exist_ok=True)

def generate_variants(img_path: Path):
    init_db()
    session = get_session()

    img = Image.open(img_path).convert("RGB")
    name = img_path.stem

    # Register original
    emb = embedder.process(img)
    sscd_vec = emb["sscd_vector"]
    phash_str = emb["phash_str"]
    sha256 = hashlib.sha256(open(img_path, 'rb').read()).hexdigest()

    # Add to FAISS index
    asset_uuid = str(uuid.uuid4())
    faiss_id = faiss_index.add(sscd_vec, asset_uuid)

    asset = Asset(
        id=asset_uuid,
        filename=img_path.name,
        sha256=sha256,
        phash=phash_str,
        faiss_id=faiss_id,
        owner="DAPS_SYSTEM"
    )
    session.add(asset)
    session.commit()
    print(f"Registered Original: {name} as {asset_uuid}")

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    variants = {}

    # T1: Exact
    variants["T1_exact"] = img.copy()

    # T2: Recompress
    t2_path = f"{name}_t2.jpg"
    img.copy().save(t2_path, quality=30)
    variants["T2_recompress"] = Image.open(t2_path)

    # T3: Crop
    crop_frac = 0.10
    left   = int(w * crop_frac)
    top    = int(h * crop_frac)
    right  = w - left
    bottom = h - top
    variants["T3_crop"] = img.crop((left, top, right, bottom))

    # T4: Filter (Color shift)
    t5_cv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    t5_cv[:, :, 0] = (t5_cv[:, :, 0] + 30) % 180
    t5_cv[:, :, 1] = np.clip(t5_cv[:, :, 1] * 1.3, 0, 255)
    t5_cv = t5_cv.astype(np.uint8)
    t5_bgr = cv2.cvtColor(t5_cv, cv2.COLOR_HSV2BGR)
    variants["T5_color"] = Image.fromarray(cv2.cvtColor(t5_bgr, cv2.COLOR_BGR2RGB))

    # Save
    for tag, variant_img in variants.items():
        out_path = QUERIES_DIR / f"{name}_{tag}.jpg"
        variant_img.convert("RGB").save(out_path, quality=92)
        print(f"  Saved variant: {out_path.name}")

    if os.path.exists(t2_path):
        os.remove(t2_path)
    session.close()

if __name__ == "__main__":
    imgs = list(ORIGINALS_DIR.glob("*.jpg")) + list(ORIGINALS_DIR.glob("*.png"))
    print(f"Processing {len(imgs)} originals...")
    for img_path in imgs:
        generate_variants(img_path)
    print("Dataset build complete.")
