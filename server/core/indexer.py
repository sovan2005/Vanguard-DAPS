import faiss
import numpy as np
import json
import logging
from pathlib import Path
from .config import cfg

logger = logging.getLogger(__name__)

class FAISSIndex:
    def __init__(self):
        self.dim        = cfg.EMBEDDING_DIM
        self.index_path = cfg.FAISS_INDEX_PATH
        self.meta_path  = self.index_path.parent / "id_map.json"
        
        self.id_map: dict[str, str] = {}
        self._next_id = 0
        self.index = None
        self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path) as f:
                data = json.load(f)
                self.id_map = data["id_map"]
                self._next_id = data["next_id"]
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
        else:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            base = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap(base)
            logger.info("Created new FAISS index")

    def add(self, embedding: np.ndarray, asset_id: str) -> int:
        import uuid
        vec = embedding.reshape(1, -1).astype(np.float32)
        # Safely assign a non-colliding integer id for faiss without needing synchronization
        faiss_id = uuid.uuid4().int % 2147483647
        self.index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))
        self.id_map[str(faiss_id)] = asset_id
        self._save()
        return faiss_id

    def search(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        
        vec = embedding.reshape(1, -1).astype(np.float32)
        k_actual = min(k, self.index.ntotal)
        D, I = self.index.search(vec, k_actual)
        
        results = []
        for score, fid in zip(D[0], I[0]):
            if fid == -1: continue
            asset_id = self.id_map.get(str(fid))
            if asset_id:
                results.append({
                    "asset_id":   asset_id,
                    "similarity": float(score),
                    "faiss_id":   int(fid),
                })
        return results

    def remove(self, faiss_id: int):
        ids = np.array([faiss_id], dtype=np.int64)
        self.index.remove_ids(ids)
        self.id_map.pop(str(faiss_id), None)
        self._save()

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump({
                "id_map":  self.id_map,
                "next_id": self._next_id,
            }, f)

    @property
    def size(self) -> int:
        return self.index.ntotal

faiss_index = FAISSIndex()
