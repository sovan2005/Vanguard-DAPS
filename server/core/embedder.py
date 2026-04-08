import torch
import numpy as np
import imagehash
from PIL import Image
from torchvision import transforms
from pathlib import Path
from .config import cfg
import logging

logger = logging.getLogger(__name__)

class SSCDEmbedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        model_path = cfg.SSCD_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"SSCD model not found at {model_path}\n"
                f"Download from: github.com/facebookresearch/sscd-copy-detection"
            )
        self.model = torch.jit.load(str(model_path))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self._loaded = True
        logger.info(f"SSCD loaded on {self.device}")

    def embed(self, img: Image.Image) -> np.ndarray:
        self._load()
        img_rgb = img.convert("RGB")
        tensor = self.preprocess(img_rgb).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)[0]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        return embedding.cpu().numpy().astype(np.float32)

class PHasher:
    def __init__(self, hash_size: int = cfg.PHASH_HASH_SIZE):
        self.hash_size = hash_size

    def compute(self, img: Image.Image) -> imagehash.ImageHash:
        return imagehash.phash(img, hash_size=self.hash_size)

    def compute_str(self, img: Image.Image) -> str:
        return str(self.compute(img))

    def hamming(self, hash_a: str, hash_b: str) -> int:
        ha = imagehash.hex_to_hash(hash_a)
        hb = imagehash.hex_to_hash(hash_b)
        return ha - hb

class DualEmbedder:
    def __init__(self):
        self.sscd   = SSCDEmbedder()
        self.phasher = PHasher()

    def process(self, img: Image.Image) -> dict:
        img_rgb = img.convert("RGB")
        return {
            "sscd_vector": self.sscd.embed(img_rgb),
            "phash_str":   self.phasher.compute_str(img_rgb),
        }

embedder = DualEmbedder()
