from sqlalchemy import Column, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class Asset(Base):
    __tablename__ = "assets"
    
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename      = Column(String, nullable=False)
    sha256        = Column(String(64), unique=True, nullable=False)
    phash         = Column(String(64), nullable=False)
    faiss_id      = Column(Integer, unique=True, nullable=False)
    owner         = Column(String, nullable=False)
    event_name    = Column(String)
    license_type  = Column(String, default="ALL_RIGHTS_RESERVED")
    registered_at = Column(DateTime, default=datetime.utcnow)
    
class Detection(Base):
    __tablename__ = "detections"
    
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_sha256     = Column(String(64))
    query_phash      = Column(String(64))
    matched_asset_id = Column(String)
    similarity_score = Column(Float)
    hybrid_score     = Column(Float)
    classification   = Column(String)
    confidence       = Column(Float)
    modification_hints = Column(Text)
    processing_ms    = Column(Integer)
    detected_at      = Column(DateTime, default=datetime.utcnow)
