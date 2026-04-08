<div align="center">
** Vanguard-DAPS **

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
    CLEAR          = "CLEAR"
    FLAG_SOFT      = "FLAG_SOFT"
    FLAG_HARD      = "FLAG_HARD"
    REQUEST_GEMINI = "REQUEST_GEMINI"
