# 🛡️ DAPSEnv — Digital Asset Protection System

> An OpenEnv-compliant RL environment where an AI agent acts as a **copyright infringement investigator**, evaluating media assets using multi-signal forensic analysis.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What It Does

An AI agent receives **forensic signal data** about a media asset and must decide:

| Action | Meaning | When to Use |
|--------|---------|-------------|
| `CLEAR` | Asset is original | Low similarity, verified source |
| `FLAG_SOFT` | Suspected copy → human review | Moderate signals, needs investigation |
| `FLAG_HARD` | Confirmed infringement → block | Strong signals, obvious copy |
| `REQUEST_GEMINI` | Deep visual analysis (costs -0.1) | Ambiguous cases only |

## 🧠 What Makes This Special

1. **9 unique task variants** — not just easy/medium/hard, but specific scenarios (adversarial decoys, AI-generated lookalikes, metadata anomalies)
2. **Confidence-weighted rewards** — high confidence on wrong answer = extra penalty. Teaches RL calibration.
3. **Forensic metadata signals** — `metadata_consistency`, `timestamp_anomaly`, `source_reputation` beyond just similarity scores
4. **Adversarial decoy task** — signals look like infringement but it's original. Tests false positive control.
5. **AI-generated lookalike detection** — cutting-edge scenario where style transfer creates similar-looking content
6. **Episode performance grading** — A+ through F based on accuracy, efficiency, and calibration

## 📊 Observation Space

| Signal | Type | Range | What It Tells You |
|--------|------|-------|-------------------|
| `sscd_score` | float | 0.0–1.0 | SSCD embedding similarity (PRIMARY signal) |
| `phash_distance` | int | 0–256 | Perceptual hash distance |
| `modification_type` | enum | 8 types | How asset was modified |
| `metadata_consistency` | float | 0.0–1.0 | How well metadata matches original |
| `source_reputation` | float | 0.0–1.0 | Trustworthiness of source |
| `timestamp_anomaly` | bool | - | Suspicious upload timing |
| `threat_level` | enum | 5 levels | Fused threat assessment |
| `gemini_verdict` | string | optional | Gemini Vision result |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                     │
│  POST /reset → start episode, get first observation │
│  POST /step  → submit action, get reward + next obs │
│  GET  /state → episode snapshot with analytics      │
│  GET  /tasks → 9 task variants with graders         │
│  GET  /info  → environment capabilities             │
│  GET  /metrics → aggregate performance analytics    │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │   DAPSEnvironment     │
         │  • 9 task generators  │
         │  • Reward function    │
         │  • Gemini simulator   │
         │  • Evidence packets   │
         │  • Episode analytics  │
         └───────────────────────┘
```

## 🚀 Quick Start

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python app.py

# In another terminal, run the baseline agent
python inference.py
```

### Docker

```bash
docker build -t dapsenv .
docker run -p 7860:7860 dapsenv
```

### HF Spaces Deployment

1. Push this repo to a Hugging Face Space (Docker SDK)
2. The server auto-starts on port 7860
3. Verify: `curl https://your-space.hf.space/health`

## 📁 Project Files

| File | Purpose |
|------|---------|
| `models.py` | Pydantic models — Action, Observation, State, EvidencePacket |
| `environment.py` | Core game engine — 9 tasks, graders, reward function |
| `app.py` | FastAPI server — all OpenEnv-compliant endpoints |
| `inference.py` | Baseline agent — rule-based + LLM fallback |
| `openenv.yaml` | OpenEnv spec declaration |
| `Dockerfile` | Container setup for HF Spaces |
| `requirements.txt` | Python dependencies |

## 🎮 Reward Structure

| Outcome | Base Reward | Notes |
|---------|-------------|-------|
| Correct decision | +1.0 × difficulty | Easy ×1.0, Medium ×1.2, Hard ×1.5 |
| Partial match | +0.5 × difficulty | FLAG_SOFT when FLAG_HARD expected |
| False positive | -0.3 | Flagged an original |
| False negative | -1.0 | Cleared an infringement |
| Gemini call | -0.1 | Cost per use |
| Confidence bonus | ±0.15 max | High confidence correct = bonus |
| 9/9 correct bonus | +1.0 | Episode completion bonus |
| 8/9 correct bonus | +0.5 | Near-perfect bonus |

## 📋 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:7860` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | - | Hugging Face token |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment URL |
| `PORT` | `7860` | Server port |

## 📜 License

MIT — built for the Meta × PyTorch OpenEnv Hackathon.
