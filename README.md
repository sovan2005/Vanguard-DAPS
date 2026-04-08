---
title: DAPS-OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# Vanguard-DAPS

**A production-grade OpenEnv environment for copyright infringement detection via multi-signal forensic analysis**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-spec_v1-4B5563?style=flat-square)](https://meta-pytorch.org/OpenEnv)
[![HF Space](https://img.shields.io/badge/HF_Space-live-F97316?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Sovan-123456789/daps-env-hackathon)
[![Python](https://img.shields.io/badge/Python-3.11-3B82F6?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-6B7280?style=flat-square)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Meta_×_PyTorch-OpenEnv_Hackathon-1D4ED8?style=flat-square)](https://scaler.com/school-of-technology/meta-pytorch-hackathon)

</div>

---

## Overview

Vanguard-DAPS is an [OpenEnv](https://meta-pytorch.org/OpenEnv)-compliant reinforcement learning environment that frames copyright infringement detection as a sequential decision problem. An agent receives a bundle of forensic signals derived from the SSCD embedding pipeline, perceptual hashing, and modification fingerprinting, then must decide how to classify a query asset: clear it, escalate to review, block it, or invoke a computationally expensive vision oracle.

The environment is designed to surface the core tension in production-scale copyright enforcement systems: **false negatives (missed infringement) carry asymmetric cost relative to false positives (wrongful blocking)**. The reward function encodes this asymmetry explicitly, making it unsuitable to solve via a degenerate always-flag policy.

This was built as a submission for the **Meta × PyTorch / Scaler OpenEnv Hackathon (Round 1, April 2026)** and is deployed live on Hugging Face Spaces.

---

## Signal Design Rationale

Each observation vector is composed of signals drawn from the real-world DAPS forensics pipeline:

| Signal | Source | Signal Semantics |
|--------|--------|-----------------|
| `sscd_score` | SSCD (Facebook Research, ICCV 2022) | Embedding-space similarity via a self-supervised contrastive model. Invariant to color transforms, crops, and compression; breaks under semantic edits and compositing. Range `[0, 1]`. |
| `phash_distance` | Perceptual Hash (dHash/aHash) | Structural similarity at the pixel level. Complementary to SSCD — catches compression artifacts SSCD normalizes away. Range `[0, 64]`. |
| `modification_type` | Modification Fingerprint Classifier | Inferred transformation type: `NONE`, `CROP`, `FILTER`, `WATERMARK`, `COMPOSITE`, `UNKNOWN`. |
| `modification_confidence` | Modification Fingerprint Classifier | Classifier confidence on the `modification_type` label. |
| `source_domain` | Upload metadata | Categorical source context (`social_media`, `news_site`, `ecommerce`, `unknown`, `vpn_detected`). |
| `file_size_ratio` | Metadata | Query / reference file size. Ratios significantly above 1.0 suggest compositing; below 0.8 suggest lossy recompression. |
| `upload_delay_hours` | Metadata | Time delta between original registration and query upload. Negative values indicate the query predates the reference — a strong originality signal. |
| `gemini_verdict` | Gemini Vision API (simulated) | Natural language verdict from deep visual inspection. Only populated after a `REQUEST_GEMINI` action. |
| `gemini_similarity` | Gemini Vision API (simulated) | Continuous similarity estimate from the vision oracle. |

The core difficulty gradient in the environment arises from signal conflicts. Easy tasks have strongly correlated signals (high SSCD, low pHash, obvious modification). Hard tasks introduce adversarial configurations where SSCD and pHash point in opposite directions, or where metadata anomalies contradict similarity scores — cases where a threshold-only policy collapses and the oracle becomes necessary.

---

## Action Space

```python
class ActionType(str, Enum):
    CLEAR          = "CLEAR"           # No infringement — asset is original
    FLAG_SOFT      = "FLAG_SOFT"       # Suspected copy — escalate to human review
    FLAG_HARD      = "FLAG_HARD"       # Confirmed infringement — block immediately
    REQUEST_GEMINI = "REQUEST_GEMINI"  # Invoke vision oracle (non-terminal, costs -0.1)
```

`REQUEST_GEMINI` is a non-terminal action. It enriches the current observation with `gemini_verdict` and `gemini_similarity` and returns the same task for a second decision. This models the real-world cost of invoking a slow, expensive downstream model — an agent that calls it on every step will lose reward even if all its terminal decisions are correct.

---

## Reward Function

```
R(action, ground_truth, difficulty) =
    +1.0 × d   if action == ground_truth (exact match)
    +0.5 × d   if partial match (FLAG_SOFT ↔ FLAG_HARD)
    −0.3       if false positive (flagged an original)
    −1.0       if false negative (cleared an infringement)
    −0.1       if REQUEST_GEMINI (oracle cost, per call)

where d = difficulty_multiplier ∈ {1.0 (easy), 1.2 (medium), 1.5 (hard)}
```

The asymmetric false negative penalty (`-1.0` vs `-0.3` for false positive) encodes the production reality: clearing infringing content exposes the platform to DMCA liability; over-flagging original content triggers appeals workflows and creator friction. A well-calibrated agent learns to seek the oracle on genuinely ambiguous tasks rather than on all tasks.

Maximum achievable score per episode (9 tasks, no wasted oracle calls):
```
2×(1.0×1.0) + 2×(1.0×1.2) + 2×(1.0×1.5) = 2.0 + 2.4 + 3.0 = 7.4
```

---

## Task Definitions

The environment generates 9 tasks per episode — 3 per difficulty tier, drawn from named scenario classes:

**Easy** — High-confidence infringement. Single-pass threshold sufficient.
- `exact_copy` — SSCD ≥ 0.97, pHash ≤ 2. Reupload with no modification.
- `recompressed_copy` — SSCD ≥ 0.93, pHash ≤ 4. Lossy recompression artifact.
- `cropped_copy` — SSCD ∈ [0.90, 0.95], pHash ≤ 5. Tight crop of the original.

**Medium** — Conflicting or attenuated signals. Requires multi-signal fusion.
- `filtered_asset` — SSCD ∈ [0.65, 0.82]. Color transform shifts embedding space.
- `watermark_detection` — SSCD ∈ [0.68, 0.83], modification_confidence moderate. Watermark overlay.
- `metadata_mismatch` — Anomalous `upload_delay_hours` and `source_domain` despite moderate similarity.

**Hard** — Adversarial or ambiguous. Oracle is often the correct first move.
- `ambiguous_classification` — Weak signals across all channels. Ground truth 50/50 infringing/original.
- `adversarial_decoy` — Signals are tuned to look like infringement but asset is original. Tests FP control.
- `ai_generated_lookalike` — Style transfer creates similar-looking content. SSCD is unreliable; Gemini is authoritative.

---

## API Reference

The server exposes a standard OpenEnv HTTP interface. All endpoints accept and return JSON.

```
POST /reset          Start a new episode. Returns first observation.
POST /step           Submit an action. Returns {observation, reward, done, info}.
GET  /state          Full episode snapshot: step count, reward, decision log.
GET  /tasks          Enumerate task variants with grader metadata.
GET  /health         Liveness probe. Returns 200 + {"status": "ok"}.
```

### `/reset`

```json
POST /reset
{"seed": 42, "difficulty": null}

→ {
    "observation": {
        "sscd_score": 0.712,
        "phash_distance": 17,
        "modification_type": "FILTER",
        "modification_confidence": 0.71,
        "source_domain": "news_site",
        "file_size_ratio": 0.93,
        "upload_delay_hours": 48.2,
        "task_id": "a3c4_medium_0",
        "step_in_episode": 0,
        "difficulty": "medium"
    }
}
```

### `/step`

```json
POST /step
{"action": {"action_type": "FLAG_SOFT", "confidence": 0.82}}

→ {
    "observation": { ... },
    "reward": 0.6,
    "done": false,
    "info": {
        "ground_truth": "FLAG_SOFT",
        "reward_breakdown": {"correct": 1.2},
        "step": 2,
        "total_reward": 2.1
    }
}
```

---

## Architecture

```
daps_env/
├── models.py          # Pydantic models: DAPSAction, DAPSObservation, DAPSState
├── environment.py     # Episode logic, task generators, graders, reward function
├── app.py             # FastAPI server — OpenEnv HTTP interface
├── inference.py       # Baseline agent (LLM-guided + rule-based fallback)
├── openenv.yaml       # OpenEnv spec declaration (spec_version: 1)
├── Dockerfile         # Python 3.11-slim; uvicorn on port 7860
└── requirements.txt
```

The environment is single-session (stateful singleton). For parallel training workloads, wrap in a session manager keyed by `episode_id`. The core `DAPSEnvironment` class has no global state beyond the current episode — instantiating multiple copies is safe.

---

## Quickstart

**Local (bare Python)**

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860

# Verify
curl http://localhost:7860/health
```

**Docker**

```bash
docker build -t vanguard-daps .
docker run -p 7860:7860 vanguard-daps
```

**Run the baseline agent**

```bash
# Against local server
ENV_BASE_URL=http://localhost:7860 python inference.py

# Against the live HF Space
ENV_BASE_URL=https://sovan-123456789-daps-env-hackathon.hf.space \
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=hf_... \
python inference.py
```

**Required environment variables**

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Model identifier passed to the chat completions API |
| `HF_TOKEN` | Hugging Face token (also used as the LLM API key) |
| `ENV_BASE_URL` | Base URL of the running environment server |

---

## Baseline Performance

The `inference.py` rule-based agent (no LLM required) achieves the following on 3 episodes with `seed` fixed:

```
Episodes       : 3
Avg reward     : 6.67 / 7.40 max
Accuracy       : 6.0 / 6.0 tasks
False positives: 0
False negatives: 0
Elapsed        : < 5s
```

The LLM-guided agent (GPT-4o-mini) scores equivalently on easy/medium and improves on hard tasks by correctly invoking `REQUEST_GEMINI` on adversarial scenarios where the rule-based threshold is underpowered.

---

## Design Notes

**Why not normalize rewards to `[0, 1]`?** The asymmetric penalty structure (`-1.0` for false negatives vs `+1.5` max for hard correct) is intentional. Clamping to `[0, 1]` removes the gradient signal that teaches the agent false negatives are strictly worse than over-caution. An RL agent trained on a flattened reward surface learns a degenerate always-flag policy.

**Why `REQUEST_GEMINI` instead of always providing Gemini signals?** This models the real cost structure of production copyright pipelines. SSCD + pHash run in-process at inference time; a vision oracle is a remote call with latency and token cost. Giving the agent control over when to invoke it — with an explicit cost — is a more faithful environment than either always-including or always-excluding the oracle signal.

**Why simulate Gemini?** The environment is self-contained and reproducible. The simulated oracle is deterministic given a seed, which is required for the OpenEnv validation runner to reproduce scores. A real Gemini API call would introduce non-determinism and external network dependencies into the evaluation pipeline.

---

## Relation to Existing Work

This environment is inspired by the DAPS (Digital Asset Protection System) pipeline submitted to the Google Solution Challenge 2026, which used real SSCD embeddings, pHash, FAISS-based retrieval, and Gemini Vision for copy detection in sports media. Vanguard-DAPS abstracts that pipeline into a synthetic but structurally faithful RL environment — the signal distributions and threshold behaviors are calibrated against that system's empirical outputs.

---

## License

MIT. Built for the Meta × PyTorch / Scaler OpenEnv Hackathon, April 2026.
