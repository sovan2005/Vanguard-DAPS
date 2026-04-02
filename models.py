"""
DAPSEnv — Digital Asset Protection System
models.py: All typed data shapes (Pydantic models)

Think of this file as defining the "vocabulary" of our environment:
  - What does the agent SEE each step? → DAPSObservation
  - What can the agent DO each step?   → DAPSAction
  - What is the overall episode state? → DAPSState

Enhanced with:
  - ThreatLevel enum for risk classification
  - Richer modification types (AI_GENERATED, RECOMPRESSION)
  - Forensic metadata signals (timestamp anomaly, source reputation)
  - EvidencePacket for structured forensic output
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# ACTION — what the agent can do
# ─────────────────────────────────────────────

class ActionType(str, Enum):
    """The four decisions an agent can make about any asset."""
    CLEAR          = "CLEAR"           # Asset is original — no infringement
    FLAG_SOFT      = "FLAG_SOFT"       # Suspected copy — queue for human review
    FLAG_HARD      = "FLAG_HARD"       # Confirmed infringement — block immediately
    REQUEST_GEMINI = "REQUEST_GEMINI"  # Invoke Gemini Vision for deeper analysis (costs -0.1)


class DAPSAction(BaseModel):
    """Single action submitted by the agent at each step."""
    action_type: ActionType = Field(
        ...,
        description="The agent's decision about the current asset."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How confident the agent is (0.0–1.0). Affects reward: high confidence on wrong answer = extra penalty."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional free-text reason (used by LLM-based agents)."
    )


# ─────────────────────────────────────────────
# THREAT LEVEL — risk classification
# ─────────────────────────────────────────────

class ThreatLevel(str, Enum):
    """Risk severity of the detected infringement."""
    BENIGN   = "BENIGN"    # No threat — original content
    LOW      = "LOW"       # Minor similarity, likely coincidence
    MEDIUM   = "MEDIUM"    # Moderate signals, needs investigation
    HIGH     = "HIGH"      # Strong signals, likely infringement
    CRITICAL = "CRITICAL"  # Near-exact copy, immediate action needed


# ─────────────────────────────────────────────
# OBSERVATION — what the agent sees each step
# ─────────────────────────────────────────────

class ModificationType(str, Enum):
    """How the asset was potentially modified."""
    NONE           = "NONE"           # No detected modification
    CROP           = "CROP"           # Cropped version
    FILTER         = "FILTER"         # Color/style filter applied
    WATERMARK      = "WATERMARK"      # Watermark added or removed
    COMPOSITE      = "COMPOSITE"      # Merged with other content
    RECOMPRESSION  = "RECOMPRESSION"  # Re-encoded with quality loss
    AI_GENERATED   = "AI_GENERATED"   # AI-generated lookalike (style transfer, etc.)
    UNKNOWN        = "UNKNOWN"        # Cannot determine


class DAPSObservation(BaseModel):
    """
    Everything the agent can observe about the current asset.

    Core signals come from the DAPS pipeline:
      - sscd_score:   SSCD embedding cosine similarity
      - phash_dist:   perceptual hash Hamming distance
      - mod_type:     modification fingerprint classifier

    Forensic metadata signals (the judge-impressing part):
      - metadata_consistency: how well metadata matches original
      - timestamp_anomaly:    suspicious upload timing
      - source_reputation:    trustworthiness of source domain
    """
    # ── Core similarity signals ──────────────────
    sscd_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="SSCD embedding cosine similarity (1.0 = identical)."
    )
    phash_distance: int = Field(
        ...,
        ge=0,
        le=256,
        description="Perceptual hash Hamming distance (0 = identical, 256 = max different)."
    )

    # ── Modification fingerprint ─────────────────
    modification_type: ModificationType = Field(
        ...,
        description="Detected type of modification applied to the asset."
    )
    modification_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the modification type classification."
    )

    # ── Metadata signals ─────────────────────────
    source_domain: str = Field(
        ...,
        description="Where the query asset was uploaded from (e.g. 'social_media', 'news_site')."
    )
    file_size_ratio: float = Field(
        ...,
        ge=0.0,
        description="Query file size / reference file size. >1.0 means query is bigger."
    )
    upload_delay_hours: float = Field(
        ...,
        description="Hours after the original asset was registered. Negative = uploaded before."
    )

    # ── Forensic metadata signals (NEW — distinguishes us) ──
    metadata_consistency: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "How consistent the query metadata is with the original. "
            "1.0 = perfectly consistent. Low values hint at tampering."
        )
    )
    timestamp_anomaly: bool = Field(
        default=False,
        description=(
            "True if the upload timestamp pattern is suspicious. "
            "E.g., uploaded within minutes of original going live = likely bot scrape."
        )
    )
    source_reputation: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Trustworthiness of source domain. "
            "1.0 = verified official channel, 0.0 = known piracy host."
        )
    )

    # ── Gemini Vision result (populated after REQUEST_GEMINI) ───
    gemini_verdict: Optional[str] = Field(
        default=None,
        description="Gemini Vision analysis result. Only populated after REQUEST_GEMINI action."
    )
    gemini_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Gemini Vision similarity score. Only populated after REQUEST_GEMINI action."
    )

    # ── Episode context ──────────────────────────
    task_id: str = Field(..., description="Unique ID for this asset check task.")
    step_in_episode: int = Field(..., description="Which step we're on (0-indexed).")
    difficulty: str = Field(..., description="Task difficulty: easy | medium | hard.")
    threat_level: ThreatLevel = Field(
        default=ThreatLevel.MEDIUM,
        description="Assessed threat level based on signal fusion."
    )


# ─────────────────────────────────────────────
# EVIDENCE PACKET — forensic output (in step info)
# ─────────────────────────────────────────────

class EvidencePacket(BaseModel):
    """
    Structured forensic evidence for each decision.
    This is what makes DAPS different from binary match/no-match tools.
    Returned inside the step info dict.
    """
    ground_truth: str = Field(description="What the correct action was.")
    agent_action: str = Field(description="What the agent chose.")
    correct: bool = Field(description="Whether the agent was correct.")
    confidence_penalty: float = Field(
        default=0.0,
        description="Extra penalty/bonus from confidence level."
    )
    signals_summary: str = Field(
        default="",
        description="Human-readable summary of why this decision was scored this way."
    )
    threat_assessment: str = Field(
        default="",
        description="Threat level assessment based on all signals."
    )
    modification_detected: str = Field(default="NONE")
    reward_breakdown: dict = Field(default_factory=dict)


# ─────────────────────────────────────────────
# STATE — full episode snapshot
# ─────────────────────────────────────────────

class DAPSState(BaseModel):
    """
    Full episode state snapshot (returned by state() endpoint).
    This is what judges/evaluators inspect to understand the episode.
    """
    episode_id: str
    current_step: int
    max_steps: int
    total_reward: float
    decisions_made: list[str] = Field(
        default_factory=list,
        description="List of action_type strings taken so far."
    )
    gemini_calls_used: int = Field(
        default=0,
        description="How many REQUEST_GEMINI calls have been made (cost tracker)."
    )
    correct_decisions: int = Field(default=0)
    false_positives: int = Field(default=0)
    false_negatives: int = Field(default=0)
    done: bool = Field(default=False)
    difficulty: str = Field(default="mixed")

    # ── Enhanced metrics ──
    accuracy: float = Field(
        default=0.0,
        description="Running accuracy: correct_decisions / total terminal actions."
    )
    gemini_efficiency: float = Field(
        default=1.0,
        description="Proportion of Gemini calls that led to correct decisions."
    )
    confidence_calibration: float = Field(
        default=0.0,
        description="How well the agent's confidence matches actual correctness."
    )
    threat_distribution: dict = Field(
        default_factory=dict,
        description="Count of how many tasks were at each threat level."
    )


# ─────────────────────────────────────────────
# STEP RESULT — what step() returns
# ─────────────────────────────────────────────

class DAPSStepResult(BaseModel):
    """Full result object returned after each step() call."""
    observation: DAPSObservation
    reward: float = Field(description="Reward for the last action.")
    done: bool = Field(description="True if the episode is over.")
    info: dict = Field(
        default_factory=dict,
        description="Debug info: evidence packet, reward breakdown, etc."
    )


# ─────────────────────────────────────────────
# EPISODE SUMMARY — end-of-episode analytics
# ─────────────────────────────────────────────

class EpisodeSummary(BaseModel):
    """Rich analytics returned when an episode ends."""
    episode_id: str
    total_reward: float
    total_steps: int
    tasks_completed: int
    correct_decisions: int
    false_positives: int
    false_negatives: int
    gemini_calls: int
    gemini_efficiency: float = Field(
        description="What fraction of Gemini calls led to correct subsequent decisions."
    )
    accuracy: float
    avg_confidence: float
    reward_per_difficulty: dict = Field(
        default_factory=dict,
        description="{'easy': total, 'medium': total, 'hard': total}"
    )
    performance_grade: str = Field(
        default="C",
        description="A/B/C/D/F grade based on overall performance."
    )
