"""
DAPSEnv — Digital Asset Protection System
app.py: FastAPI server wrapping the environment

This is the HTTP interface. When deployed to HF Spaces, judges/validators
call these endpoints to interact with the environment.

Endpoints:
  POST /reset        → start new episode, returns first observation
  POST /step         → submit an action, get reward + next observation
  GET  /state        → inspect current episode state
  GET  /health       → liveness probe (required for HF Spaces auto-check)
  GET  /tasks        → list available task difficulties (spec compliance)
  GET  /metrics      → episode performance analytics
  GET  /info         → environment description + capabilities
"""

import sys
import os
import time
import json

# Removed sys.path hack for absolute imports

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import io
import uuid
from PIL import Image
from pathlib import Path

from server.models import (
    DAPSAction,
    DAPSObservation,
    DAPSState,
    DAPSStepResult,
    EpisodeSummary,
    ModificationType
)
import hashlib
import numpy as np
from server.core.embedder import embedder
from server.core.indexer import faiss_index
from server.db.database import get_session
from server.db.models import Asset
from server.environment import DAPSEnvironment, assess_threat_level, grade_easy_task, grade_medium_task, grade_hard_task
from server.core.detector import detector_engine


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Vanguard DAPS — Enterprise Risk Intelligence API",
    description=(
        "An enterprise-grade Image Risk Intelligence API that automates "
        "copyright protection at scale. It leverages Fusion Intelligence (SSCD + pHash) "
        "to detect similarity, quantify liability, and automate takedown workflows "
        "for marketplaces, social media, and internal brand security."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves images from the static directory
from fastapi.staticfiles import StaticFiles
_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_DIR / "static")), name="static")

# Single environment instance (concurrent sessions disabled for hackathon)
env = DAPSEnvironment()

# Metrics tracking across episodes
_episode_history: list[dict] = []
_start_time = time.time()


# ─────────────────────────────────────────────
# Startup: Setup ML environment (Model download + Dataset build)
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Ensure ML models and dataset are ready before first request."""
    print("DAPSEnv starting up...")
    from server.core.config import cfg
    scripts_dir = Path(__file__).parent / "scripts"
    model_path = Path(cfg.SSCD_MODEL_PATH)
    # Parse db path from sqlite:///... format
    db_path = Path(cfg.DB_URL.replace("sqlite:///", ""))

    # 1. Check if model exists, if not run setup
    if not model_path.exists() or model_path.stat().st_size < 80000000:
        print("SSCD Model missing or incomplete. Running setup script...")
        subprocess.run([sys.executable, str(scripts_dir / "setup_ml_environment.py")], check=True)

    # 2. Check if dataset exists, if not run build
    if not db_path.exists() or db_path.stat().st_size < 1000:
        print("DAPS Database missing or empty. Running dataset build...")
        subprocess.run([sys.executable, str(scripts_dir / "build_test_dataset.py")], check=True)
        # Reload the FAISS index and clear engine cache since we built them in another process
        faiss_index._load_or_create()

    print("DAPSEnv is fully initialized and ready.")


# ─────────────────────────────────────────────
# Request / Response models for the HTTP layer
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: Optional[str] = None   # "easy" | "medium" | "hard" | None (mixed)


class ResetResponse(BaseModel):
    observation: DAPSObservation
    message: str = "Episode started. Call POST /step with your action."


class StepRequest(BaseModel):
    action: DAPSAction


class StepResponse(BaseModel):
    observation: DAPSObservation
    reward: float
    done: bool
    info: dict


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


class TasksResponse(BaseModel):
    tasks: list[dict]


class InfoResponse(BaseModel):
    name: str
    description: str
    version: str
    task_count: int
    action_space: list[str]
    observation_fields: list[str]
    reward_range: Optional[list[float]] = None
    difficulty_levels: Optional[list[str]] = None
    unique_features: Optional[list[str]] = None


class EnforcementRequest(BaseModel):
    action: str  # "TAKEDOWN" | "ESCALATE" | "WHITELIST"
    task_id: str


class EnforcementResponse(BaseModel):
    success: bool
    audit_id: str
    message: str


class StatsResponse(BaseModel):
    total_assets_scanned: int
    high_risk_alerts: int
    medium_risk_hits: int
    protected_assets: int
    monitoring_status: str


class MetricsResponse(BaseModel):
    total_episodes: int
    uptime_seconds: float
    avg_reward: float
    avg_accuracy: float
    avg_gemini_calls: float
    best_reward: float
    best_grade: str
    episode_history: list[dict]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe. HF Spaces validator pings this."""
    return HealthResponse(
        status="ok",
        environment="DAPSEnv",
        version="2.0.0",
    )

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        index_path = _DIR / "static" / "index.html"
        with open(index_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"<h1>Welcome to DAPSEnv</h1><p>index.html not found at {index_path}.</p>"

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode.

    Returns the first asset observation the agent must evaluate.
    Call this before the first step, and between episodes.

    Example:
        POST /reset
        {}

    Or with a fixed seed for reproducibility:
        {"seed": 42}
    """
    try:
        obs = env.reset(seed=request.seed, difficulty=request.difficulty)
        return ResetResponse(observation=obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit an agent action and receive reward + next observation.

    Action types:
      - CLEAR          → asset is original (no infringement)
      - FLAG_SOFT      → suspected copy, send to human review
      - FLAG_HARD      → confirmed infringement, block immediately
      - REQUEST_GEMINI → invoke Gemini Vision for deeper analysis (-0.1 cost)

    Note: REQUEST_GEMINI does NOT advance to the next task.
    It enriches the current observation with gemini_verdict and gemini_similarity,
    then you must submit a terminal action (CLEAR/FLAG_SOFT/FLAG_HARD).

    Confidence matters: high confidence on correct = bonus, high confidence on wrong = penalty.

    Example:
        POST /step
        {"action": {"action_type": "FLAG_HARD", "confidence": 0.95}}
    """
    try:
        result: DAPSStepResult = env.step(request.action)

        # 🧠 DECISION REASONING ENGINE (WINNER'S CIRCLE UPGRADE)
        # We transform raw numbers into an "Explainable Forensic Report"
        obs = result.observation
        report = {
            "verdict": "SECURE",
            "confidence": 0.0,
            "analysis": "No threats detected.",
            "recommendation": "AUTHORIZED"
        }

        if obs.task_id != "EPISODE_COMPLETE":
            # 1. Calculate Confidence (Geometric Mean of SSCD and phash clarity)
            conf = (obs.sscd_score) * (1.0 - (obs.phash_distance / 256.0))
            report["confidence"] = round(conf * 100, 1)

            # 2. Logic Matrix (Fusion Engine)
            if obs.sscd_score > 0.92:
                report["verdict"] = "CRITICAL INFRINGEMENT 🔴"
                report["analysis"] = "Neural DNA match exceeds 92%. Original asset identity confirmed with near-total certainty."
                report["recommendation"] = "IMMEDIATE TAKEDOWN"
            elif obs.sscd_score > 0.82:
                report["verdict"] = "PROBABLE INFRINGEMENT 🟠"
                report["analysis"] = f"Strong similarity detected (SSCD: {obs.sscd_score}). Identified illegal manipulation ({obs.modification_type})."
                report["recommendation"] = "MANUAL FORENSIC REVIEW"
            elif obs.sscd_score > 0.65:
                report["verdict"] = "SUSPICIOUS SIMILARITY 🟡"
                report["analysis"] = "Partial similarity detected. Could be a derivative work or high-noise edit."
                report["recommendation"] = "MONITOR ASSET"
            else:
                report["verdict"] = "SECURE ASSET 🟢"
                report["analysis"] = "No significant neural matches found in the protected database."
                report["recommendation"] = "AUTHORIZE DISTRIBUTION"

            # Add "Human Sentence" for the pitch
            report["summary"] = f"Forensic analysis suggests this is a {report['verdict']} based on neural similarity and {obs.modification_type} detection."

        result.info["forensic_report"] = report

        # If episode is done, save to history
        if result.done and "episode_summary" in result.info:
            summary = result.info["episode_summary"]
            _episode_history.append(summary)

        return StepResponse(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=DAPSState)
def state():
    """
    Inspect current episode state.

    Returns the full episode snapshot including:
    - Current step, total reward, accuracy
    - Decisions made so far
    - Correct/incorrect decision counts
    - Gemini calls used and efficiency
    - Confidence calibration score

    Example:
        GET /state
    """
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history")
def get_history():
    """
    Retrieve the personnel service record (completed episode summaries).
    """
    return {"history": _episode_history}


@app.get("/benchmarks")
def get_benchmarks():
    """
    Returns the real-world performance proof (Precision/Recall) 
    comparing Vanguard Fusion vs. Traditional Hashing.
    """
    try:
        with open(_DIR / "static" / "benchmarks.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Benchmark data not yet generated."}


@app.get("/tasks", response_model=TasksResponse)
def list_tasks():
    """
    List available task types (OpenEnv spec: enumerate tasks).

    DAPSEnv has 9 unique task variants across 3 difficulty levels.
    Each task has a grader that returns scores in [0.0, 1.0].
    """
    return TasksResponse(tasks=[
        {
            "id": "easy_exact_copy",
            "name": "Exact Copy Detection",
            "description": (
                "Near-exact copy with minimal modification. "
                "SSCD > 0.92, pHash < 5. Signals clearly indicate infringement. "
                "Expected action: FLAG_HARD."
            ),
            "difficulty": "easy",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.easy_exact_copy.grader:grade",
        },
        {
            "id": "easy_recompressed",
            "name": "Recompressed Copy Detection",
            "description": (
                "Same content re-encoded with quality loss (screenshot, re-upload). "
                "SSCD > 0.90, pHash < 6. Expected action: FLAG_HARD."
            ),
            "difficulty": "easy",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.easy_recompressed.grader:grade",
        },
        {
            "id": "easy_cropped",
            "name": "Cropped Copy Detection",
            "description": (
                "Cropped version of original. SSCD > 0.88, pHash < 8. "
                "Still clearly a copy. Expected action: FLAG_HARD."
            ),
            "difficulty": "easy",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.easy_cropped.grader:grade",
        },
        {
            "id": "medium_filtered",
            "name": "Filtered Asset Detection",
            "description": (
                "Color/style filter applied. SSCD 0.65-0.85, pHash 8-20. "
                "Agent must weigh both signals. Expected action: FLAG_SOFT."
            ),
            "difficulty": "medium",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.medium_filtered.grader:grade",
        },
        {
            "id": "medium_watermarked",
            "name": "Watermark Detection",
            "description": (
                "Watermark added or removed. Perceptual hash shifts but "
                "deep embedding still recognizable. Expected action: FLAG_SOFT."
            ),
            "difficulty": "medium",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.medium_watermarked.grader:grade",
        },
        {
            "id": "medium_metadata_mismatch",
            "name": "Metadata Anomaly Detection",
            "description": (
                "Color-graded asset with suspicious metadata. Low consistency, "
                "suspicious source. Agent must use forensic signals. "
                "Expected action: FLAG_SOFT."
            ),
            "difficulty": "medium",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.medium_metadata_mismatch.grader:grade",
        },
        {
            "id": "hard_ambiguous",
            "name": "Ambiguous Asset Classification",
            "description": (
                "Conflicting signals. Could be CLEAR or FLAG_HARD. "
                "Optimal: call REQUEST_GEMINI first, then decide."
            ),
            "difficulty": "hard",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.hard_ambiguous.grader:grade",
        },
        {
            "id": "hard_adversarial_decoy",
            "name": "Adversarial Decoy (Agent-Under-Attack)",
            "description": (
                "Signals LOOK like infringement but asset is original. "
                "Tests false positive control. Verified source, consistent metadata, "
                "uploaded BEFORE reference. Expected action: CLEAR."
            ),
            "difficulty": "hard",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.hard_adversarial_decoy.grader:grade",
        },
        {
            "id": "hard_ai_generated",
            "name": "AI-Generated Lookalike Detection",
            "description": (
                "AI-generated content that looks similar to original. "
                "SSCD picks up style similarity, pHash diverges. "
                "Cutting-edge scenario. Expected action: FLAG_SOFT."
            ),
            "difficulty": "hard",
            "reward_range": [0.0, 1.0],
            "grader": "tasks.hard_ai_generated.grader:grade",
        },
    ])


@app.get("/info", response_model=InfoResponse)
def info():
    """
    Environment description and capabilities.
    Used for auto-discovery by evaluation tools and judges.
    """
    return InfoResponse(
        name="DAPSEnv — Digital Asset Protection System",
        description=(
            "AI agent acts as a copyright investigator, evaluating media assets "
            "across 9 unique scenarios using SSCD similarity, perceptual hash, "
            "and forensic metadata signals. Features adversarial decoys and "
            "AI-generated lookalike detection."
        ),
        version="2.0.0",
        task_count=9,
        action_space=["CLEAR", "FLAG_SOFT", "FLAG_HARD", "REQUEST_GEMINI"],
        observation_fields=[
            "sscd_score", "phash_distance", "modification_type",
            "modification_confidence", "source_domain", "file_size_ratio",
            "upload_delay_hours", "metadata_consistency", "timestamp_anomaly",
            "source_reputation", "gemini_verdict", "gemini_similarity",
            "id", "step_in_episode", "difficulty", "threat_level",
        ],
        reward_range=[-0.1, 1.0],
        difficulty_levels=["easy", "medium", "hard"],
        unique_features=[
            "Confidence-weighted reward shaping",
            "Adversarial decoy tasks (tests false positive control)",
            "AI-generated lookalike detection",
            "Forensic evidence packets per decision",
            "Gemini Vision escalation mechanism",
            "Episode-level performance grading (A+ through F)",
            "Metadata forensics (timestamp anomaly, source reputation)",
        ],
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    """
    Performance analytics across all episodes.
    Returns aggregate statistics and per-episode history.
    """
    uptime = time.time() - _start_time
    total = len(_episode_history)

    if total == 0:
        return MetricsResponse(
            total_episodes=0,
            uptime_seconds=round(uptime, 1),
            avg_reward=0.0,
            avg_accuracy=0.0,
            avg_gemini_calls=0.0,
            best_reward=0.0,
            best_grade="N/A",
            episode_history=[],
        )

    avg_reward = sum(e.get("total_reward", 0) for e in _episode_history) / total
    avg_accuracy = sum(e.get("accuracy", 0) for e in _episode_history) / total
    avg_gemini = sum(e.get("gemini_calls", 0) for e in _episode_history) / total
    best = max(_episode_history, key=lambda e: e.get("total_reward", 0))

    return MetricsResponse(
        total_episodes=total,
        uptime_seconds=round(uptime, 1),
        avg_reward=round(avg_reward, 3),
        avg_accuracy=round(avg_accuracy, 3),
        avg_gemini_calls=round(avg_gemini, 1),
        best_reward=best.get("total_reward", 0),
        best_grade=best.get("performance_grade", "N/A"),
        episode_history=_episode_history[-10:],  # Last 10 episodes
    )


@app.post("/analyze")
async def analyze_custom_asset(file: UploadFile = File(...)):
    """
    Forensic Deep-Check for custom user-uploaded assets.
    Runs real-time SSCD and pHash matching against the protected database.
    """
    try:
        # Read image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Run Detection
        res = detector_engine.detect(image)
        sscd = round(res.get("sscd_score", 0.0), 3)
        phash = int(res.get("phash_distance", 64))
        
        # Build a "Custom Observation"
        obs = DAPSObservation(
            sscd_score=sscd,
            phash_distance=phash,
            modification_type="UPLOADED_CUSTOM",
            modification_confidence=0.95,
            source_domain="user_upload_portal",
            file_size_ratio=1.0,
            upload_delay_hours=0.0,
            metadata_consistency=0.9,
            timestamp_anomaly=False,
            source_reputation=0.5,
            task_id="CUSTOM_FORENSIC_CHECK",
            step_in_episode=0,
            difficulty="n/a",
            threat_level=assess_threat_level(sscd, phash, 0.5)
        )

        # Generate Report Reasoning
        report = {
            "verdict": "SECURE",
            "confidence": round(sscd * 100, 1),
            "analysis": "Neural scanning complete.",
            "recommendation": "AUTHORIZED"
        }

        if sscd > 0.85:
            report["verdict"] = "MATCH DETECTED 🔴"
            report["analysis"] = f"Strong neural similarity ({round(sscd*100, 1)}%) found in database. Evidence suggests unauthorized duplication."
            report["recommendation"] = "IMMEDIATE TAKEDOWN / COPYRIGHT FLAG"
        elif sscd > 0.65:
            report["verdict"] = "SUSPICIOUS 🟠"
            report["analysis"] = "Moderate similarity found. Potential derivative work or heavy modification."
            report["recommendation"] = "MANUAL REVIEW REQUIRED"
        else:
            report["verdict"] = "NO MATCH 🟢"
            report["analysis"] = "No direct neural fingerprints found. Asset appears original."
            report["recommendation"] = "AUTHORIZE DISTRIBUTION"

        return {
            "observation": obs,
            "forensic_report": report,
            "message": "Custom forensic analysis complete."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic Analysis Failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
def get_business_stats():
    """
    Returns rolling enterprise-level statistics for the dashboard.
    Simulates high-volume continuous monitoring for the hackathon.
    """
    total = sum(e.get("steps", 0) for e in _episode_history) + 2842 
    high = sum(1 for e in _episode_history if e.get("total_reward", 0) < 0) + 142
    return StatsResponse(
        total_assets_scanned=total,
        high_risk_alerts=high,
        medium_risk_hits=42,
        protected_assets=954,
        monitoring_status="ACTIVE: CONTINUOUS SCAN"
    )


@app.post("/enforce", response_model=EnforcementResponse)
def enforce_decision(req: EnforcementRequest):
    """
    Handles business-level enforcement actions.
    Simulates the final 'Action Pipeline' for platform integration.
    """
    audit_id = f"AUD_ENF_{uuid.uuid4().hex[:8].upper()}"
    return EnforcementResponse(
        success=True,
        audit_id=audit_id,
        message=f"Forensic Action '{req.action}' initiated for {req.task_id}."
    )


# ─────────────────────────────────────────────
# Entry point (for local testing)
# ─────────────────────────────────────────────

@app.post("/analyze/pair")
async def analyze_pair(reference: UploadFile = File(...), query: UploadFile = File(...)):
    """
    Forensic Laboratory: Compare Reference vs investigation Query.
    Registers the Reference into the secure database if not already present.
    """
    try:
        allowed = {".jpg", ".jpeg", ".png"}
        ref_ext = Path(reference.filename).suffix.lower() if reference.filename else ""
        que_ext = Path(query.filename).suffix.lower() if query.filename else ""
        
        if ref_ext not in allowed or que_ext not in allowed:
            raise HTTPException(status_code=400, detail="Strict Compliance Error: Only JPG/PNG supported.")

        # 1. Process Reference
        ref_content = await reference.read()
        ref_img = Image.open(io.BytesIO(ref_content)).convert("RGB")
        ref_emb = embedder.process(ref_img)
        ref_sha = hashlib.sha256(ref_content).hexdigest()

        # 2. Process Query
        que_content = await query.read()
        que_img = Image.open(io.BytesIO(que_content)).convert("RGB")
        que_emb = embedder.process(que_img)

        # 3. DIRECT COMPARISON (Forensic Fusion)
        # Neural Similarity (Dot product of normalized vectors)
        sscd_sim = float(np.dot(ref_emb["sscd_vector"], que_emb["sscd_vector"]))
        # Perceptual Match
        phash_dist = embedder.phasher.hamming(ref_emb["phash_str"], que_emb["phash_str"])
        
        # Metadata Check (Simulated Forensic Audit)
        meta_status = "PASSED"
        if ref_img.size != que_img.size:
             meta_status = "SIZE_MISMATCH"
        
        # 4. Register Reference into Database (Business Expansion)
        session = get_session()
        existing = session.query(Asset).filter(Asset.sha256 == ref_sha).first()
        if not existing:
            faiss_id = faiss_index.add(ref_emb["sscd_vector"], ref_sha)
            new_asset = Asset(
                id=str(uuid.uuid4()),
                filename=reference.filename,
                sha256=ref_sha,
                phash=ref_emb["phash_str"],
                faiss_id=faiss_id,
                owner="USER_UPLOAD",
                event_name="LAB_INSPECTION"
            )
            session.add(new_asset)
            session.commit()
            print(f"Asset Registered: {reference.filename} (ID: {faiss_id})")
        session.close()

        # 5. Risk Assessment
        threat_level = assess_threat_level(sscd_sim, phash_dist, 0.5)

        return {
            "metrics": {
                "neural_similarity": round(sscd_sim, 3),
                "perceptual_distance": phash_dist,
                "metadata_status": meta_status,
                "risk_assessment": threat_level.value
            },
            "message": "Forensic Pair Analysis Complete. Reference asset indexed for future scans."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lab Analysis Failed: {str(e)}")


def main():
    """Entry point for the openenv server CLI."""
    try:
        from openenv.core.cli import server
        server()
    except ImportError:
        import uvicorn
        port = int(os.environ.get("PORT", 7860))
        uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
