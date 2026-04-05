import random
import uuid
from typing import Optional
from pathlib import Path
from PIL import Image

from models import (
    ActionType,
    DAPSAction,
    DAPSObservation,
    DAPSState,
    DAPSStepResult,
    EvidencePacket,
    EpisodeSummary,
    ModificationType,
    ThreatLevel,
)

from core.detector import detector_engine

QUERIES_DIR = Path("data/queries")

# ─────────────────────────────────────────────────────────────
# THREAT LEVEL ASSESSMENT
# ─────────────────────────────────────────────────────────────

def assess_threat_level(sscd: float, phash: int, source_rep: float) -> ThreatLevel:
    if sscd >= 0.90 and phash <= 5:
        return ThreatLevel.CRITICAL
    elif sscd >= 0.75 and phash <= 15:
        return ThreatLevel.HIGH
    elif sscd >= 0.60 and phash <= 25:
        return ThreatLevel.MEDIUM
    elif sscd >= 0.45:
        return ThreatLevel.LOW
    else:
        return ThreatLevel.BENIGN

# ─────────────────────────────────────────────────────────────
# ML DETECTOR HELPER
# ─────────────────────────────────────────────────────────────
def run_ml_detector(variant_suffix: str):
    """Picks a random test image of the specified variant and runs actual PyTorch SSCD/pHash inference!"""
    imgs = list(QUERIES_DIR.glob(f"*{variant_suffix}.jpg"))
    if not imgs:
        # fallback if strict variant missing
        imgs = list(QUERIES_DIR.glob("*_T1_exact.jpg"))
        if not imgs:
            return 0.95, 2 # absolute fallback if dataset missing

    img_path = random.choice(imgs)
    img = Image.open(img_path)
    res = detector_engine.detect(img)
    sscd = round(res.get("sscd_score", 0.0), 3)
    phash = res.get("phash_distance", 64)
    return sscd, phash

# ─────────────────────────────────────────────────────────────
# TASK DEFINITIONS
# ─────────────────────────────────────────────────────────────

def _make_easy_task_exact_copy(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T1_exact")
    source_rep = round(random.uniform(0.1, 0.4), 2)
    obs = DAPSObservation(
        sscd_score=sscd, phash_distance=phash, modification_type=ModificationType.CROP,
        modification_confidence=0.9, source_domain="social_media", file_size_ratio=0.95,
        upload_delay_hours=5.0, metadata_consistency=0.4, timestamp_anomaly=True,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="easy", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ActionType.FLAG_HARD

def _make_easy_task_recompressed(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T2_recompress")
    source_rep = round(random.uniform(0.2, 0.5), 2)
    obs = DAPSObservation(
        sscd_score=sscd, phash_distance=phash, modification_type=ModificationType.RECOMPRESSION,
        modification_confidence=0.9, source_domain="messaging_app", file_size_ratio=0.7,
        upload_delay_hours=2.0, metadata_consistency=0.5, timestamp_anomaly=False,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="easy", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ActionType.FLAG_HARD

def _make_easy_task_cropped(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T3_crop")
    source_rep = round(random.uniform(0.15, 0.45), 2)
    obs = DAPSObservation(
        sscd_score=sscd, phash_distance=phash, modification_type=ModificationType.CROP,
        modification_confidence=0.85, source_domain="blog", file_size_ratio=0.8,
        upload_delay_hours=12.0, metadata_consistency=0.3, timestamp_anomaly=False,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="easy", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ActionType.FLAG_HARD

def _make_medium_task_filtered(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T5_color")
    source_rep = round(random.uniform(0.3, 0.6), 2)
    obs = DAPSObservation(
        sscd_score=sscd, phash_distance=phash, modification_type=ModificationType.FILTER,
        modification_confidence=0.7, source_domain="news_site", file_size_ratio=1.0,
        upload_delay_hours=20.0, metadata_consistency=0.5, timestamp_anomaly=False,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="medium", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ActionType.FLAG_SOFT

def _make_medium_task_watermarked(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T5_color") # fallback to filter for moderate signals
    if sscd > 0.8: sscd -= 0.1 # augment simulated watermark impact
    phash += 5                 # augment
    source_rep = round(random.uniform(0.3, 0.7), 2)
    obs = DAPSObservation(
        sscd_score=sscd, phash_distance=phash, modification_type=ModificationType.WATERMARK,
        modification_confidence=0.7, source_domain="ecommerce", file_size_ratio=1.1,
        upload_delay_hours=50.0, metadata_consistency=0.6, timestamp_anomaly=False,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="medium", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ActionType.FLAG_SOFT

def _make_medium_task_metadata_mismatch(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T2_recompress")
    if sscd > 0.8: sscd -= 0.15 
    phash += 2
    source_rep = round(random.uniform(0.1, 0.3), 2)
    obs = DAPSObservation(
        sscd_score=sscd, phash_distance=phash, modification_type=ModificationType.FILTER,
        modification_confidence=0.6, source_domain="unknown_aggregator", file_size_ratio=1.2,
        upload_delay_hours=1.0, metadata_consistency=0.2, timestamp_anomaly=True,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="medium", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ActionType.FLAG_SOFT

def _make_hard_task_ambiguous(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T5_color")
    is_infringement = random.choice([True, False])

    if is_infringement:
        if sscd > 0.7: sscd -= 0.2
        phash += 15
        source_rep = 0.3
        ground_truth = ActionType.FLAG_HARD
    else:
        if sscd > 0.6: sscd -= 0.3
        phash += 20
        source_rep = 0.8
        ground_truth = ActionType.CLEAR

    obs = DAPSObservation(
        sscd_score=round(sscd,3), phash_distance=int(phash), modification_type=ModificationType.COMPOSITE,
        modification_confidence=0.5, source_domain="unknown", file_size_ratio=2.0,
        upload_delay_hours=10.0, metadata_consistency=0.3, timestamp_anomaly=True,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="hard", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ground_truth

def _make_hard_task_adversarial_decoy(task_id: str) -> tuple[DAPSObservation, ActionType]:
    # Decoy has high similarity but it is actually the source!
    sscd, phash = run_ml_detector("_T2_recompress")
    source_rep = 0.95
    obs = DAPSObservation(
        sscd_score=sscd, phash_distance=phash, modification_type=ModificationType.NONE,
        modification_confidence=0.3, source_domain="official_channel", file_size_ratio=1.0,
        upload_delay_hours=-100.0, metadata_consistency=0.95, timestamp_anomaly=False,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="hard", threat_level=ThreatLevel.LOW
    )
    return obs, ActionType.CLEAR

def _make_hard_task_ai_generated(task_id: str) -> tuple[DAPSObservation, ActionType]:
    sscd, phash = run_ml_detector("_T5_color")
    if sscd > 0.8: sscd -= 0.1
    phash += 12
    source_rep = 0.3
    obs = DAPSObservation(
        sscd_score=round(sscd,3), phash_distance=int(phash), modification_type=ModificationType.AI_GENERATED,
        modification_confidence=0.7, source_domain="ai_art_platform", file_size_ratio=1.2,
        upload_delay_hours=48.0, metadata_consistency=0.3, timestamp_anomaly=False,
        source_reputation=source_rep, task_id=task_id, step_in_episode=0,
        difficulty="hard", threat_level=assess_threat_level(sscd, phash, source_rep)
    )
    return obs, ActionType.FLAG_SOFT

def compute_reward(action: DAPSAction, ground_truth: ActionType, difficulty: str, gemini_called: bool) -> tuple[float, dict]:
    info = {
        "ground_truth": ground_truth.value, "agent_action": action.action_type.value,
        "difficulty": difficulty, "reward_breakdown": {}, "confidence_used": action.confidence,
    }
    if action.action_type == ActionType.REQUEST_GEMINI:
        info["reward_breakdown"]["gemini_cost"] = -0.1
        return -0.1, info

    difficulty_multiplier = {"easy": 1.0, "medium": 1.2, "hard": 1.5}[difficulty]
    confidence_modifier = 0.0

    if action.action_type in (ActionType.FLAG_HARD, ActionType.FLAG_SOFT) and ground_truth in (ActionType.FLAG_HARD, ActionType.FLAG_SOFT):
        if action.action_type == ground_truth:
            r = 1.0 * difficulty_multiplier
            info["correct"] = True
            confidence_modifier = 0.15 * (action.confidence - 0.5) * 2
        else:
            r = 0.5 * difficulty_multiplier
            info["correct"] = False
        r += confidence_modifier
        info["reward_breakdown"]["confidence_modifier"] = round(confidence_modifier, 4)
        return round(r, 3), info

    if action.action_type == ActionType.CLEAR and ground_truth == ActionType.CLEAR:
        r = 1.0 * difficulty_multiplier
        info["correct"] = True
        confidence_modifier = 0.15 * (action.confidence - 0.5) * 2
        r += confidence_modifier
        info["reward_breakdown"]["confidence_modifier"] = round(confidence_modifier, 4)
        return round(r, 3), info

    if action.action_type == ActionType.CLEAR and ground_truth != ActionType.CLEAR:
        r = -1.0
        info["correct"] = False
        confidence_modifier = -0.15 * action.confidence
        r += confidence_modifier
        info["reward_breakdown"]["confidence_modifier"] = round(confidence_modifier, 4)
        return round(r, 3), info

    if action.action_type in (ActionType.FLAG_HARD, ActionType.FLAG_SOFT) and ground_truth == ActionType.CLEAR:
        r = -0.3
        info["correct"] = False
        confidence_modifier = -0.15 * action.confidence
        r += confidence_modifier
        info["reward_breakdown"]["confidence_modifier"] = round(confidence_modifier, 4)
        return round(r, 3), info
    return 0.0, info

def simulate_gemini_call(obs: DAPSObservation, ground_truth: ActionType) -> dict:
    if ground_truth in (ActionType.FLAG_HARD, ActionType.FLAG_SOFT):
        return {"verdict": "SUSPECTED_COPY", "gemini_similarity": 0.85, "note": "Similar to reference."}
    else:
        return {"verdict": "LIKELY_ORIGINAL", "gemini_similarity": 0.25, "note": "Independent creation."}

EASY_GENERATORS = [_make_easy_task_exact_copy, _make_easy_task_recompressed, _make_easy_task_cropped]
MEDIUM_GENERATORS = [_make_medium_task_filtered, _make_medium_task_watermarked, _make_medium_task_metadata_mismatch]
HARD_GENERATORS = [_make_hard_task_ambiguous, _make_hard_task_adversarial_decoy, _make_hard_task_ai_generated]

class DAPSEnvironment:
    def __init__(self):
        self._state = None
        self._task_queue = []
        self._current_obs = None
        self._current_ground_truth = None
        self._awaiting_gemini = False
        self._confidence_history = []
        self._gemini_correct_after = 0
        self._reward_by_difficulty = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        self._total_terminal_actions = 0

    def reset(self, seed=None, difficulty=None) -> DAPSObservation:
        if seed is not None: random.seed(seed)
        episode_id = str(uuid.uuid4())[:8]

        # Use partials or delayed execution to avoid running 9 ML inferences on startup
        task_blueprints = []
        
        # Collect all generators with their expected ID
        for i in range(3):
            task_blueprints.append((EASY_GENERATORS[i], f"{episode_id}_easy_{i}"))
            task_blueprints.append((MEDIUM_GENERATORS[i], f"{episode_id}_med_{i}"))
            task_blueprints.append((HARD_GENERATORS[i], f"{episode_id}_hard_{i}"))

        random.shuffle(task_blueprints)
        self._task_queue = task_blueprints

        self._state = DAPSState(
            episode_id=episode_id, 
            current_step=0, 
            max_steps=len(task_blueprints), 
            total_reward=0.0, 
            difficulty=difficulty or "mixed"
        )
        self._awaiting_gemini = False
        self._confidence_history = []
        self._gemini_correct_after = 0
        self._reward_by_difficulty = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        self._total_terminal_actions = 0

        # Lazy execution of the first task only
        gen_fn, tid = self._task_queue.pop(0)
        self._current_obs, self._current_ground_truth = gen_fn(tid)
        self._current_obs = self._current_obs.model_copy(update={"step_in_episode": 0})
        return self._current_obs

    def step(self, action: DAPSAction) -> DAPSStepResult:
        if self._state is None or self._current_obs is None: raise RuntimeError("Call reset() before step().")
        info = {}
        if action.action_type == ActionType.REQUEST_GEMINI:
            self._state.gemini_calls_used += 1
            self._awaiting_gemini = True
            gemini_result = simulate_gemini_call(self._current_obs, self._current_ground_truth)
            enriched_obs = self._current_obs.model_copy(update={
                "gemini_verdict": gemini_result["verdict"], "gemini_similarity": gemini_result["gemini_similarity"],
            })
            self._current_obs = enriched_obs
            reward = -0.1
            self._state.total_reward = round(self._state.total_reward + reward, 3)
            info = {"gemini_result": gemini_result, "reward_breakdown": {"gemini_cost": -0.1}}
            return DAPSStepResult(observation=enriched_obs, reward=reward, done=False, info=info)

        self._total_terminal_actions += 1
        reward, info = compute_reward(action, self._current_ground_truth, self._current_obs.difficulty, self._awaiting_gemini)
        is_correct = info.get("correct", False)
        self._confidence_history.append((action.confidence, is_correct))
        if self._awaiting_gemini and is_correct: self._gemini_correct_after += 1

        diff = self._current_obs.difficulty
        self._reward_by_difficulty[diff] = round(self._reward_by_difficulty.get(diff, 0.0) + reward, 3)

        evidence = EvidencePacket(
            ground_truth=self._current_ground_truth.value, agent_action=action.action_type.value,
            correct=is_correct, confidence_penalty=info.get("reward_breakdown", {}).get("confidence_modifier", 0.0),
            signals_summary=f"SSCD={self._current_obs.sscd_score:.3f} pHash={self._current_obs.phash_distance}",
            threat_assessment=self._current_obs.threat_level.value, modification_detected=self._current_obs.modification_type.value,
            reward_breakdown=info.get("reward_breakdown", {}),
        )
        info["evidence"] = evidence.model_dump()

        self._state.total_reward = round(self._state.total_reward + reward, 3)
        self._state.current_step += 1
        self._state.decisions_made.append(action.action_type.value)

        if is_correct: self._state.correct_decisions += 1
        elif info.get("reward_breakdown", {}).get("false_positive"): self._state.false_positives += 1
        elif info.get("reward_breakdown", {}).get("false_negative"): self._state.false_negatives += 1

        self._state.accuracy = round(self._state.correct_decisions / self._total_terminal_actions, 3)
        if self._state.gemini_calls_used > 0:
            self._state.gemini_efficiency = round(self._gemini_correct_after / self._state.gemini_calls_used, 3)

        if self._confidence_history:
            cal = sum(1.0 - abs(conf - (1.0 if correct else 0.0)) for conf, correct in self._confidence_history) / len(self._confidence_history)
            self._state.confidence_calibration = round(cal, 3)

        self._awaiting_gemini = False
        done = False
        if self._task_queue:
            # Lazy execution of the NEXT task
            gen_fn, tid = self._task_queue.pop(0)
            self._current_obs, self._current_ground_truth = gen_fn(tid)
            self._current_obs = self._current_obs.model_copy(update={"step_in_episode": self._state.current_step})
            next_obs = self._current_obs
        else:
            done = True
            self._state.done = True
            if self._state.correct_decisions >= 8:
                bonus = 1.0
                info["episode_bonus"] = bonus
                self._state.total_reward = round(self._state.total_reward + bonus, 3)
            elif self._state.correct_decisions >= 7:
                bonus = 0.5
                info["episode_bonus"] = bonus
                self._state.total_reward = round(self._state.total_reward + bonus, 3)

            avg_conf = sum(c for c, _ in self._confidence_history) / len(self._confidence_history) if self._confidence_history else 0.0
            
            # --- Professional Forensic Grading System ---
            acc = self._state.accuracy
            fn = self._state.false_negatives
            rew = self._state.total_reward
            cal = self._state.confidence_calibration
            
            if acc >= 1.0 and fn == 0 and rew >= 10.0: grade = "S (Elite)"
            elif acc >= 0.95 and fn == 0: grade = "A+ (Expert)"
            elif acc >= 0.88 and fn == 0: grade = "A (Pro)"
            elif acc >= 0.78 and fn <= 1: grade = "B (Competent)"
            elif acc >= 0.65 and fn <= 2: grade = "C (Junior)"
            elif acc >= 0.50: grade = "D (Novice)"
            else: grade = "F (Security Risk)"
            
            # Critical override: multiple false negatives always fail a professional system
            if fn >= 3: grade = "F (Unsafe)"

            info["episode_summary"] = EpisodeSummary(
                episode_id=self._state.episode_id, total_reward=self._state.total_reward, total_steps=self._state.current_step,
                tasks_completed=self._total_terminal_actions, correct_decisions=self._state.correct_decisions,
                false_positives=self._state.false_positives, false_negatives=self._state.false_negatives,
                gemini_calls=self._state.gemini_calls_used, gemini_efficiency=self._state.gemini_efficiency,
                accuracy=self._state.accuracy, avg_confidence=round(avg_conf, 3), reward_per_difficulty=self._reward_by_difficulty,
                performance_grade=grade
            ).model_dump()
            next_obs = self._current_obs.model_copy(update={"task_id": "EPISODE_COMPLETE"})

        info["step"] = self._state.current_step
        info["total_reward"] = self._state.total_reward

        return DAPSStepResult(observation=next_obs, reward=reward, done=done, info=info)

    def state(self) -> DAPSState:
        if self._state is None: raise RuntimeError("Call reset() before state().")
        return self._state
