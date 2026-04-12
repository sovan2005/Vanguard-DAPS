import sys
import os
import traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import uuid
import hashlib

# Fix relative imports
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from server.core.embedder import embedder
    from server.core.indexer import faiss_index
    from server.db.models import Asset
    from server.db.database import get_session, init_db
except Exception as e:
    print(f"CRITICAL ERROR: Failed to import internal modules: {e}")
    traceback.print_exc()
    sys.exit(1)

ORIGINALS_DIR = BASE_DIR / "data" / "originals"
QUERIES_DIR   = BASE_DIR / "data" / "queries"
QUERIES_DIR.mkdir(parents=True, exist_ok=True)

def generate_variants(img_path: Path):
    try:
        init_db()
        session = get_session()

        img = Image.open(img_path).convert("RGB")
        name = img_path.stem

        print(f"Processing image: {name} (Size: {img.size})")

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
        print(f"  Registered Original: {asset_uuid}")

        variants = {}

        # T1: Exact
        variants["T1_exact"] = img.copy()

        # T2: Recompress
        t2_out = QUERIES_DIR / f"{name}_T2_recompress.jpg"
        img.copy().save(t2_out, "JPEG", quality=30)
        variants["T2_recompress"] = Image.open(t2_out)

        # T3: Crop
        w, h = img.size
        crop_frac = 0.15
        left   = int(w * crop_frac)
        top    = int(h * crop_frac)
        right  = w - left
        bottom = h - top
        variants["T3_crop"] = img.crop((left, top, right, bottom))

        # T5: Color/Style (Replaces CV2-based T5)
        # Using PIL Enhance for better compatibility in headless Docker
        enhancer = ImageEnhance.Color(img)
        img_color = enhancer.enhance(1.8)
        enhancer2 = ImageEnhance.Contrast(img_color)
        variants["T5_color"] = enhancer2.enhance(1.2)

        # Save all
        for tag, variant_img in variants.items():
            if tag == "T2_recompress": continue # Already saved
            out_path = QUERIES_DIR / f"{name}_{tag}.jpg"
            variant_img.convert("RGB").save(out_path, "JPEG", quality=92)
            print(f"    Saved variant: {out_path.name}")

        session.close()
    except Exception as e:
        print(f"ERROR: Failed to process {img_path.name}: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        imgs = list(ORIGINALS_DIR.glob("*.jpg")) + list(ORIGINALS_DIR.glob("*.png"))
        print(f"Found {len(imgs)} base images in {ORIGINALS_DIR}")
        
        if not imgs:
            print("WARNING: No images found to process!")
            
        for img_path in imgs:
            generate_variants(img_path)
        
        print("Dataset build complete.")
    except Exception as e:
        print(f"BUILD FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
