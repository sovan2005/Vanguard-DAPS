import numpy as np
import time
import logging
from PIL import Image
from .config import cfg
from .embedder import embedder
from .indexer import faiss_index
from ..db.database import get_session
from ..db.models import Asset

logger = logging.getLogger(__name__)

def hybrid_score(phash_hamming: int, cosine_sim: float) -> float:
    phash_score = 1.0 - (phash_hamming / 256.0)
    return (cfg.PHASH_WEIGHT * phash_score + cfg.SSCD_WEIGHT  * cosine_sim)

class DetectionEngine:
    def __init__(self):
        self.embedder = embedder
        self.index    = faiss_index

    def detect(self, img: Image.Image) -> dict:
        t_start = time.time()
        emb = self.embedder.process(img)
        query_sscd   = emb["sscd_vector"]
        query_phash  = emb["phash_str"]

        candidates = self.index.search(query_sscd, k=5)
        if not candidates:
            return {"match": False, "sscd_score": 0.0, "phash_distance": 256}

        best = None
        best_hybrid = -1.0

        session = get_session()
        for candidate in candidates:
            asset = session.query(Asset).filter(Asset.id == candidate["asset_id"]).first()
            if asset is None: continue

            hamming = self.embedder.phasher.hamming(query_phash, asset.phash)
            cosine = candidate["similarity"]
            
            hs = hybrid_score(hamming, cosine)
            if hs > best_hybrid:
                best_hybrid = hs
                best = {
                    "asset":   asset,
                    "hamming": hamming,
                    "cosine":  cosine,
                    "hybrid":  hs,
                }
        session.close()

        if not best:
             return {"match": False, "sscd_score": 0.0, "phash_distance": 256}
             
        return {
             "match": True,
             "sscd_score": best["cosine"],
             "phash_distance": best["hamming"],
             "hybrid_score": best["hybrid"],
             "matched_asset_id": best["asset"].id,
             "matched_filename": best["asset"].filename
        }

detector_engine = DetectionEngine()
