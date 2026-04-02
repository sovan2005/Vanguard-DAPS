"""
DAPSEnv — inference.py
Baseline agent for the Digital Asset Protection System environment.

Hackathon requirement: Must use OpenAI client with env vars:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face token (used for HF Spaces access)

Runtime constraint: Must complete in < 20 minutes on 2 vCPU / 8GB RAM.

Enhanced agent with:
  - Expert forensic investigator LLM prompt with decision trees
  - Multi-step strategy: REQUEST_GEMINI on hard/ambiguous tasks
  - Episode memory: tracks patterns across steps
  - Metadata-aware rule-based fallback
  - Better baseline scores (actual learning/adaptation)
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# Config from environment variables (required by hackathon)
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# The deployed environment URL (HF Spaces). Falls back to localhost.
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Number of full episodes to run for the baseline evaluation
NUM_EPISODES = 3


# ─────────────────────────────────────────────
# LLM client (OpenAI-compatible)
# ─────────────────────────────────────────────

llm_client = OpenAI(
    api_key=HF_TOKEN or "placeholder",
    base_url=API_BASE_URL,
)


# ─────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────

def env_reset(seed=None):
    payload = {}
    if seed is not None:
        payload["seed"] = seed
    resp = requests.post(f"{ENV_BASE_URL}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["observation"]


def env_step(action_type: str, confidence: float = 1.0, reason: str = None):
    payload = {
        "action": {
            "action_type": action_type,
            "confidence": confidence,
            "reason": reason,
        }
    }
    resp = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_state():
    resp = requests.get(f"{ENV_BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Episode memory — tracks decisions within an episode
# ─────────────────────────────────────────────

class EpisodeMemory:
    """Tracks patterns across steps to help the agent adapt."""
    def __init__(self):
        self.steps = []
        self.gemini_used_on_current = False

    def record(self, obs: dict, action: str, reward: float):
        self.steps.append({
            "difficulty": obs.get("difficulty", "unknown"),
            "action": action,
            "reward": reward,
            "sscd": obs.get("sscd_score", 0),
            "phash": obs.get("phash_distance", 0),
        })

    def reset_gemini(self):
        self.gemini_used_on_current = False

    def summary(self) -> str:
        if not self.steps:
            return "No previous steps."
        correct = sum(1 for s in self.steps if s["reward"] > 0)
        total = len(self.steps)
        return f"Step {total}: {correct}/{total} correct so far."


# ─────────────────────────────────────────────
# Rule-based fallback agent (Enhanced)
# Uses SSCD + pHash + metadata signals for smarter decisions.
# ─────────────────────────────────────────────

def rule_based_decision(obs: dict, memory: EpisodeMemory) -> tuple[str, float, str]:
    """
    Metadata-aware threshold agent. Returns (action_type, confidence, reason).
    Optimized for the real PyTorch SSCD and 256-bit FAISS Index pipeline.
    """
    sscd = obs["sscd_score"]
    phash = obs["phash_distance"]
    gemini_verdict = obs.get("gemini_verdict")
    meta_consist = obs.get("metadata_consistency", 0.5)
    source_rep = obs.get("source_reputation", 0.5)
    timestamp_anom = obs.get("timestamp_anomaly", False)
    mod_type = obs.get("modification_type", "UNKNOWN")
    difficulty = obs.get("difficulty", "medium")

    # 1. Gemini Verdict Override
    if gemini_verdict:
        if gemini_verdict == "SUSPECTED_COPY":
            return "FLAG_HARD", 0.95, "Gemini confirmed suspected copy"
        else:
            return "CLEAR", 0.90, "Gemini found likely original"

    # 2. Check for adversarial decoy signals BEFORE simple thresholding
    if (source_rep >= 0.75 and
        meta_consist >= 0.80 and
        obs.get("upload_delay_hours", 0) < -5.0 and
        mod_type == "NONE"):
        return "CLEAR", 0.82, (
            f"Adversarial decoy pattern: SourceRep={source_rep:.2f} "
            f"MetaConsist={meta_consist:.2f} UploadBefore=True"
        )

    # 3. AI-generated detection
    if mod_type == "AI_GENERATED":
        return "FLAG_SOFT", 0.78, f"AI-generated content detected, mod_conf={obs.get('modification_confidence', 0):.2f}"

    # 4. EASY: High confidence infringement 
    # Adjusted for realistic PyTorch SSCD outputs and 256-bit pHash scale.
    if sscd >= 0.82 and phash <= 140:
        return "FLAG_HARD", 0.95, f"SSCD={sscd:.2f} pHash={phash} → obvious copy"

    # 5. MEDIUM: Moderate signals
    if sscd >= 0.58 and phash <= 180:
        if meta_consist < 0.45 or source_rep < 0.4:
            return "FLAG_SOFT", 0.80, (
                f"SSCD={sscd:.2f} pHash={phash} + suspicious metadata "
                f"(consistency={meta_consist:.2f}, rep={source_rep:.2f})"
            )
        return "FLAG_SOFT", 0.72, f"SSCD={sscd:.2f} pHash={phash} → suspected copy"

    # 6. CLEAR original
    if sscd < 0.52 and phash > 150 and source_rep > 0.5:
        return "CLEAR", 0.80, f"SSCD={sscd:.2f} pHash={phash} SourceRep={source_rep:.2f} → likely original"

    # 7. HARD ambiguous → use Gemini if not yet called
    if difficulty == "hard" and not memory.gemini_used_on_current:
        memory.gemini_used_on_current = True
        return "REQUEST_GEMINI", 0.60, f"Ambiguous: SSCD={sscd:.2f} pHash={phash}, requesting Gemini"

    # 8. Final fallback
    if sscd >= 0.55:
        return "FLAG_SOFT", 0.55, f"Borderline: SSCD={sscd:.2f} pHash={phash}, erring towards flag"
    return "CLEAR", 0.50, f"Insufficient evidence: SSCD={sscd:.2f} pHash={phash}"


# ─────────────────────────────────────────────
# LLM-based agent (Enhanced prompt)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert digital forensics investigator specializing in copyright infringement detection.

You will receive signal data about a media asset and must decide what action to take.

## Actions
- **CLEAR**: The asset is original. No infringement detected.
- **FLAG_SOFT**: Suspected copy — queue for human review. Use when signals are suggestive but not conclusive.
- **FLAG_HARD**: Confirmed infringement — block immediately. Use only when signals strongly indicate a copy.
- **REQUEST_GEMINI**: Invoke Gemini Vision for deeper visual analysis. Costs -0.1 reward. Use only on truly ambiguous cases.

## Signals You Receive
- **sscd_score**: SSCD embedding similarity (1.0 = identical). This is the PRIMARY signal.
- **phash_distance**: Perceptual hash Hamming distance (0 = identical, 256 = completely different).
- **modification_type**: What kind of modification was applied (CROP, FILTER, WATERMARK, COMPOSITE, RECOMPRESSION, AI_GENERATED, NONE, UNKNOWN).
- **modification_confidence**: How confident the detector is about the modification type.
- **source_domain**: Where the asset came from.
- **metadata_consistency**: How consistent query metadata is with original (1.0 = perfect match). LOW values = tampering.
- **source_reputation**: Trustworthiness of source (1.0 = verified, 0.0 = known piracy). HIGH values suggest legitimacy.
- **timestamp_anomaly**: True if upload timing is suspicious (immediate scrape pattern).
- **upload_delay_hours**: Hours after original. NEGATIVE = uploaded BEFORE original → likely legitimate.
- **threat_level**: Pre-assessed threat level based on signal fusion.
- **gemini_verdict**: Gemini Vision result (if REQUEST_GEMINI was already called).

## Decision Rules (Priority Order)
1. If gemini_verdict exists → use it directly (SUSPECTED_COPY → FLAG_HARD, LIKELY_ORIGINAL → CLEAR).
2. If source_reputation >= 0.75 AND metadata_consistency >= 0.80 AND upload_delay < -5.0 AND modification = NONE → CLEAR (adversarial decoy pattern).
3. If SSCD >= 0.82 AND pHash <= 140 → FLAG_HARD (obvious copy).
4. If modification_type = AI_GENERATED → FLAG_SOFT (needs human review).
5. If SSCD >= 0.58 AND pHash <= 180 → FLAG_SOFT (moderate signals).
6. If SSCD < 0.52 AND phash > 150 AND source_reputation > 0.5 → CLEAR.
7. If ambiguous (rules don't clearly apply) AND difficulty = hard → REQUEST_GEMINI.
8. Use confidence to express certainty (0.5 = unsure, 0.95 = very confident).

## Confidence Guidelines
- 0.90+ only for clear-cut SSCD > 0.82 cases or post-Gemini decisions
- 0.70-0.89 for moderate signal alignment
- 0.50-0.69 for ambiguous cases where you're guessing
- NEVER report 1.0 confidence

Respond with ONLY a JSON object:
{
  "action_type": "CLEAR" | "FLAG_SOFT" | "FLAG_HARD" | "REQUEST_GEMINI",
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation referencing specific signals"
}

No other text. Just the JSON."""


def llm_decision(obs: dict, memory: EpisodeMemory) -> tuple[str, float, str]:
    """Ask the LLM to decide. Falls back to rule-based if it fails."""
    obs_text = json.dumps({
        "sscd_score": obs["sscd_score"],
        "phash_distance": obs["phash_distance"],
        "modification_type": obs["modification_type"],
        "modification_confidence": obs["modification_confidence"],
        "source_domain": obs["source_domain"],
        "file_size_ratio": obs["file_size_ratio"],
        "upload_delay_hours": obs["upload_delay_hours"],
        "metadata_consistency": obs.get("metadata_consistency", 0.5),
        "source_reputation": obs.get("source_reputation", 0.5),
        "timestamp_anomaly": obs.get("timestamp_anomaly", False),
        "threat_level": obs.get("threat_level", "MEDIUM"),
        "gemini_verdict": obs.get("gemini_verdict"),
        "gemini_similarity": obs.get("gemini_similarity"),
        "difficulty": obs["difficulty"],
        "episode_context": memory.summary(),
    }, indent=2)

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Evaluate this asset:\n{obs_text}"},
            ],
            max_tokens=250,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        action_type = parsed.get("action_type", "CLEAR")
        confidence = float(parsed.get("confidence", 0.8))
        reason = parsed.get("reason", "")

        valid_actions = {"CLEAR", "FLAG_SOFT", "FLAG_HARD", "REQUEST_GEMINI"}
        if action_type not in valid_actions:
            raise ValueError(f"Invalid action: {action_type}")

        return action_type, confidence, reason

    except Exception as e:
        print(f"  [LLM fallback] {e} — using rule-based agent")
        return rule_based_decision(obs, memory)


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

def run_episode(episode_num: int, seed: int = None, use_llm: bool = True) -> dict:
    """Run a single complete episode. Returns episode stats."""
    print(f"\n{'='*60}")
    print(f"Episode {episode_num + 1}  (seed={seed})")
    print(f"{'='*60}")

    obs = env_reset(seed=seed)
    memory = EpisodeMemory()
    episode_reward = 0.0
    step_count = 0
    gemini_calls = 0
    decisions = []
    done = False

    while not done:
        step_count += 1
        print(f"\n  Step {step_count} | Task: {obs['task_id']} | Diff: {obs['difficulty']}")
        print(f"    SSCD={obs['sscd_score']:.3f}  pHash={obs['phash_distance']}  "
              f"Mod={obs['modification_type']}  "
              f"MetaC={obs.get('metadata_consistency', '?')}  "
              f"SrcRep={obs.get('source_reputation', '?')}  "
              f"Threat={obs.get('threat_level', '?')}")

        # Get agent decision
        if use_llm and API_BASE_URL and API_BASE_URL != ENV_BASE_URL:
            action_type, confidence, reason = llm_decision(obs, memory)
        else:
            action_type, confidence, reason = rule_based_decision(obs, memory)

        print(f"    → {action_type}  (conf={confidence:.2f})  reason: {reason}")

        if action_type == "REQUEST_GEMINI":
            gemini_calls += 1

        # Submit action
        result = env_step(action_type, confidence, reason)
        reward = result["reward"]
        done = result["done"]

        # Record in memory
        memory.record(obs, action_type, reward)

        obs = result["observation"]
        episode_reward += reward
        decisions.append(action_type)
        print(f"    Reward: {reward:+.3f}  | Episode total: {episode_reward:+.3f}")

        # Show evidence if available
        if "evidence" in result.get("info", {}):
            ev = result["info"]["evidence"]
            correct_str = "✅" if ev.get("correct") else "❌"
            print(f"    {correct_str} Ground truth: {ev.get('ground_truth')}")

        # Show episode summary if done
        if done and "episode_summary" in result.get("info", {}):
            summary = result["info"]["episode_summary"]
            print(f"\n  📊 Episode Summary:")
            print(f"     Grade: {summary.get('performance_grade', '?')}")
            print(f"     Accuracy: {summary.get('accuracy', 0):.0%}")
            print(f"     Gemini efficiency: {summary.get('gemini_efficiency', 0):.0%}")
            print(f"     Reward by difficulty: {summary.get('reward_per_difficulty', {})}")

        if done:
            print(f"\n  Episode complete!")
            # Reset Gemini state for next task
            memory.reset_gemini()
            break

        # Reset Gemini tracking when moving to next task (non-Gemini action)
        if action_type != "REQUEST_GEMINI":
            memory.reset_gemini()

    # Get final state
    final_state = env_state()

    stats = {
        "episode": episode_num + 1,
        "total_reward": round(episode_reward, 3),
        "steps": step_count,
        "gemini_calls": gemini_calls,
        "correct_decisions": final_state.get("correct_decisions", 0),
        "false_positives": final_state.get("false_positives", 0),
        "false_negatives": final_state.get("false_negatives", 0),
        "accuracy": final_state.get("accuracy", 0),
        "confidence_calibration": final_state.get("confidence_calibration", 0),
        "decisions": decisions,
    }

    print(f"\n  Stats: reward={stats['total_reward']}  correct={stats['correct_decisions']}/9  "
          f"FP={stats['false_positives']}  FN={stats['false_negatives']}  "
          f"acc={stats['accuracy']:.0%}")
    return stats


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAPSEnv — Digital Asset Protection System")
    print("Baseline Inference Agent v2.0")
    print("=" * 60)
    print(f"  ENV_BASE_URL : {ENV_BASE_URL}")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  NUM_EPISODES : {NUM_EPISODES}")

    # Check environment is reachable
    try:
        resp = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        health = resp.json()
        print(f"\n  Environment: {health.get('environment', '?')} v{health.get('version', '?')}")
        print(f"  Health: OK ✅")
    except Exception as e:
        print(f"\n  ERROR: Cannot reach environment at {ENV_BASE_URL}: {e}")
        sys.exit(1)

    # Show environment info if available
    try:
        resp = requests.get(f"{ENV_BASE_URL}/info", timeout=10)
        if resp.status_code == 200:
            info = resp.json()
            print(f"  Tasks: {info.get('task_count', '?')} variants")
            print(f"  Features: {', '.join(info.get('unique_features', [])[:3])}...")
    except Exception:
        pass

    # Run episodes
    all_stats = []
    start = time.time()

    for i in range(NUM_EPISODES):
        stats = run_episode(
            episode_num=i,
            seed=42 + i,
            use_llm=bool(API_BASE_URL and API_BASE_URL != ENV_BASE_URL),
        )
        all_stats.append(stats)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    avg_reward = sum(s["total_reward"] for s in all_stats) / len(all_stats)
    avg_correct = sum(s["correct_decisions"] for s in all_stats) / len(all_stats)
    avg_accuracy = sum(s["accuracy"] for s in all_stats) / len(all_stats)
    total_fp = sum(s["false_positives"] for s in all_stats)
    total_fn = sum(s["false_negatives"] for s in all_stats)

    print(f"Episodes        : {NUM_EPISODES}")
    print(f"Avg reward      : {avg_reward:.3f}")
    print(f"Avg correct     : {avg_correct:.1f} / 9 tasks")
    print(f"Avg accuracy    : {avg_accuracy:.0%}")
    print(f"False positives : {total_fp} (total)")
    print(f"False negatives : {total_fn} (total)")
    print(f"Elapsed         : {elapsed:.1f}s")
    print(f"\nAll episode rewards: {[s['total_reward'] for s in all_stats]}")

    # Write scores to file (for automated evaluation)
    scores = {
        "avg_reward": round(avg_reward, 4),
        "avg_accuracy": round(avg_accuracy, 4),
        "episodes": all_stats,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores written to baseline_scores.json")

    # Exit 0 on success (required for automated evaluation)
    sys.exit(0)


if __name__ == "__main__":
    main()
