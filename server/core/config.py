import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class Config:
    # Model
    SSCD_MODEL_PATH = BASE_DIR / "models" / "sscd_disc_mixup.torchscript.pt"
    EMBEDDING_DIM   = 512
    
    # pHash
    PHASH_HASH_SIZE = 16          # 256-bit hash
    PHASH_THRESHOLD = 20          # Hamming distance gate
    
    # FAISS
    FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss_index" / "daps.index"
    
    # Hybrid scoring weights
    PHASH_WEIGHT    = 0.20
    SSCD_WEIGHT     = 0.80
    
    # Classification thresholds
    ORIGINAL_THRESH = 0.90
    MODIFIED_THRESH = 0.75
    HEAVY_THRESH    = 0.60
    
    # DB
    DB_URL = f"sqlite:///{BASE_DIR / 'daps.db'}"
    
    # API
    MAX_FILE_SIZE_MB = 10
    ALLOWED_TYPES    = {"image/jpeg", "image/png", "image/webp"}

cfg = Config()
