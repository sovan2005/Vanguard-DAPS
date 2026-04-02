import { useState } from "react";

const COLORS = {
  bg: "#0a0a0f",
  surface: "#12121a",
  border: "#1e1e2e",
  accent: "#e8ff47",
  accentDim: "rgba(232,255,71,0.12)",
  accentBorder: "rgba(232,255,71,0.3)",
  red: "#ff4757",
  redDim: "rgba(255,71,87,0.12)",
  blue: "#4fc3f7",
  blueDim: "rgba(79,195,247,0.12)",
  purple: "#b388ff",
  purpleDim: "rgba(179,136,255,0.12)",
  orange: "#ffab40",
  orangeDim: "rgba(255,171,64,0.12)",
  green: "#69ff47",
  greenDim: "rgba(105,255,71,0.12)",
  text: "#e8e8f0",
  muted: "#6b6b8a",
  dim: "#3a3a55",
};

const style = {
  wrap: {
    background: COLORS.bg,
    color: COLORS.text,
    fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
    minHeight: "100vh",
    padding: "0",
    overflowX: "hidden",
  },
  header: {
    borderBottom: `1px solid ${COLORS.border}`,
    padding: "28px 40px 24px",
    background: COLORS.surface,
    display: "flex",
    alignItems: "center",
    gap: "20px",
  },
  tag: (color) => ({
    background: `rgba(${color},0.12)`,
    border: `1px solid rgba(${color},0.35)`,
    color: `rgb(${color})`,
    fontSize: "10px",
    padding: "3px 10px",
    borderRadius: "2px",
    letterSpacing: "2px",
    textTransform: "uppercase",
    fontWeight: "600",
  }),
  nav: {
    display: "flex",
    borderBottom: `1px solid ${COLORS.border}`,
    background: COLORS.surface,
    overflowX: "auto",
  },
  navBtn: (active) => ({
    padding: "14px 24px",
    background: "none",
    border: "none",
    borderBottom: active ? `2px solid ${COLORS.accent}` : "2px solid transparent",
    color: active ? COLORS.accent : COLORS.muted,
    cursor: "pointer",
    fontSize: "11px",
    letterSpacing: "1.5px",
    textTransform: "uppercase",
    fontWeight: active ? "700" : "400",
    whiteSpace: "nowrap",
    transition: "color 0.2s",
  }),
  main: {
    padding: "36px 40px",
    maxWidth: "1100px",
    margin: "0 auto",
  },
  section: {
    marginBottom: "48px",
  },
  sectionTitle: {
    fontSize: "10px",
    letterSpacing: "3px",
    color: COLORS.muted,
    textTransform: "uppercase",
    marginBottom: "20px",
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  divLine: {
    flex: 1,
    height: "1px",
    background: COLORS.border,
  },
  card: (accentColor = COLORS.border) => ({
    background: COLORS.surface,
    border: `1px solid ${accentColor}`,
    borderRadius: "4px",
    padding: "24px",
    marginBottom: "16px",
  }),
  grid2: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "16px",
  },
  grid3: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: "16px",
  },
  h2: {
    fontSize: "22px",
    fontWeight: "700",
    letterSpacing: "-0.5px",
    marginBottom: "6px",
    color: COLORS.text,
  },
  h3: (color = COLORS.text) => ({
    fontSize: "13px",
    fontWeight: "700",
    letterSpacing: "1px",
    textTransform: "uppercase",
    color,
    marginBottom: "12px",
  }),
  p: {
    fontSize: "13px",
    lineHeight: "1.8",
    color: "#9898b8",
    marginBottom: "10px",
  },
  mono: (color = COLORS.accent) => ({
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "12px",
    color,
    background: `rgba(232,255,71,0.06)`,
    padding: "2px 6px",
    borderRadius: "2px",
  }),
  step: (color) => ({
    display: "flex",
    gap: "16px",
    alignItems: "flex-start",
    marginBottom: "16px",
  }),
  stepNum: (color) => ({
    width: "28px",
    height: "28px",
    minWidth: "28px",
    borderRadius: "2px",
    background: color,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "10px",
    fontWeight: "700",
    color: COLORS.bg,
    letterSpacing: "1px",
  }),
  arrow: {
    textAlign: "center",
    color: COLORS.dim,
    fontSize: "18px",
    margin: "4px 0",
    letterSpacing: "2px",
  },
  pill: (color, bg) => ({
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    background: bg,
    border: `1px solid ${color}`,
    color: color,
    fontSize: "11px",
    padding: "4px 12px",
    borderRadius: "20px",
    marginRight: "8px",
    marginBottom: "8px",
    fontWeight: "600",
  }),
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "12px",
  },
  th: {
    background: COLORS.border,
    color: COLORS.muted,
    padding: "10px 14px",
    textAlign: "left",
    fontSize: "10px",
    letterSpacing: "2px",
    textTransform: "uppercase",
    fontWeight: "600",
  },
  td: {
    padding: "10px 14px",
    borderBottom: `1px solid ${COLORS.border}`,
    color: "#9898b8",
    verticalAlign: "top",
  },
  codeBlock: {
    background: "#0d0d15",
    border: `1px solid ${COLORS.border}`,
    borderRadius: "4px",
    padding: "20px",
    fontSize: "12px",
    lineHeight: "1.9",
    color: "#9898b8",
    overflowX: "auto",
    whiteSpace: "pre",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  flowBox: (color, bgColor) => ({
    background: bgColor,
    border: `1px solid ${color}`,
    borderRadius: "4px",
    padding: "14px 20px",
    marginBottom: "4px",
    fontSize: "12px",
    color: color,
    fontWeight: "600",
    letterSpacing: "0.5px",
  }),
  badge: (color) => ({
    background: color,
    color: COLORS.bg,
    fontSize: "9px",
    fontWeight: "800",
    padding: "2px 8px",
    borderRadius: "2px",
    letterSpacing: "1.5px",
    textTransform: "uppercase",
    marginLeft: "8px",
  }),
};

// ─── OODA TAB ───────────────────────────────────────────────────────────────
function OodaTab() {
  const phases = [
    {
      id: "OBSERVE",
      color: COLORS.blue,
      bg: COLORS.blueDim,
      icon: "◉",
      subtitle: "Gather all intelligence before building",
      timeframe: "Day 1–2",
      items: [
        { label: "Market Signal", text: "Sports IP piracy is a $28B/yr problem. Enterprise tools (Redflag AI, Piracy Guard) serve only top-tier leagues. Zero accessible open tools exist for smaller federations." },
        { label: "Technical Signal", text: "SSCD outperforms CLIP by 48% on DISC2021 copy detection benchmark. Hybrid pHash + deep embedding is the 2025 academic consensus for robustness." },
        { label: "Competitor Signal", text: "All existing systems are binary match/no-match. None classify modification type. None provide explainable evidence packets for legal use." },
        { label: "Gap Signal", text: "No sports-domain-specific augmentation test suite exists anywhere in literature or commercial products." },
        { label: "User Signal", text: "Rights teams need actionable forensic output — not a percentage. They need: what was copied, how it was modified, provenance chain for DMCA." },
      ],
    },
    {
      id: "ORIENT",
      color: COLORS.purple,
      bg: COLORS.purpleDim,
      icon: "◈",
      subtitle: "Frame the problem correctly — define your unique angle",
      timeframe: "Day 2–3",
      items: [
        { label: "Core Reframe", text: "Don't build a copy detector. Build a forensic evidence engine. Copy detection is the mechanism. Actionable IP enforcement intelligence is the product." },
        { label: "Model Decision", text: "Replace clip-ViT-B-32 as primary with SSCD (purpose-built for copy detection). Use CLIP as a secondary semantic signal for the multi-class layer." },
        { label: "Architecture Decision", text: "Two-stage pipeline: pHash fast-filter (milliseconds) → SSCD deep embedding (robustness). Hybrid score = 0.3×pHash + 0.7×SSCD cosine similarity." },
        { label: "Differentiation Lock-in", text: "Modification type classification is the winning feature. No existing tool in academia or industry returns: crop_detected + color_shift + overlay_present as structured output." },
        { label: "Evaluation Strategy", text: "Use DISC2021 benchmark + custom sports augmentation matrix (8 test cases) to prove claims with hard numbers, not vague assertions." },
      ],
    },
    {
      id: "DECIDE",
      color: COLORS.orange,
      bg: COLORS.orangeDim,
      icon: "◆",
      subtitle: "Commit to irreversible architectural decisions",
      timeframe: "Day 3–4",
      items: [
        { label: "Decision 1 — Stack", text: "Backend: FastAPI + FAISS + SSCD. Frontend: React (not Streamlit — Streamlit signals prototype, React signals product). Database: SQLite for hackathon, schema-ready for Postgres." },
        { label: "Decision 2 — Evidence Packet Schema", text: "Lock the output schema: {similarity_score, classification, confidence, modification_hints[], matched_asset_id, processing_time_ms}. All modules must produce output compatible with this schema." },
        { label: "Decision 3 — Demo Flow", text: "Demo is: Upload original → Upload modified (with broadcast overlay + crop) → System returns 84% Modified Reuse + modification fingerprint. This is the winning moment. Build backward from this." },
        { label: "Decision 4 — Scope Cuts", text: "Cut video pipeline from hackathon scope. Cut real-time streaming. Cut model training. Ship: embedding + hybrid scoring + multi-class classification + evidence API + React dashboard. Nothing else." },
        { label: "Decision 5 — Metric Thresholds", text: "Calibrated thresholds: >90% Original, 75–90% Modified, 60–75% Heavy Modification, <60% Unauthorized. Test against DISC2021 and sports augmentation suite before finalizing." },
      ],
    },
    {
      id: "ACT",
      color: COLORS.green,
      bg: COLORS.greenDim,
      icon: "▶",
      subtitle: "Execute in disciplined sprints, validate fast",
      timeframe: "Day 4–7",
      items: [
        { label: "Sprint 1 — Core Engine (Day 4)", text: "SSCD embedding pipeline working. pHash integrated. FAISS index operational. Hybrid score formula validated on 10 test images. Unit test: exact copy must score >95%." },
        { label: "Sprint 2 — Classification Layer (Day 5)", text: "Multi-class threshold classifier. Modification hint detector (crop, color shift, overlay, composite). Evidence packet JSON builder. Validate against full 8-case sports augmentation matrix." },
        { label: "Sprint 3 — API Layer (Day 5–6)", text: "FastAPI: POST /register, POST /detect, GET /report/{id}. Provenance metadata store. Response time target: <500ms for detection. Load test with 100 registered assets." },
        { label: "Sprint 4 — Dashboard (Day 6–7)", text: "React UI: Asset registration panel, detection panel, evidence report card with modification badges, similarity meter. Must look production-grade — judges evaluate presentation quality heavily." },
        { label: "Feedback Loop", text: "After every sprint, run demo flow (original → modified → detection). If demo breaks, stop and fix before moving forward. Demo integrity > feature count." },
      ],
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: "32px" }}>
        <div style={style.h2}>OODA LOOP — Strategic Decision Framework</div>
        <p style={style.p}>
          Observe → Orient → Decide → Act. Applied to DAPS as a competitive intelligence and execution framework, not just a project plan.
        </p>
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginTop: "12px" }}>
          <span style={style.pill(COLORS.blue, COLORS.blueDim)}>◉ Observe: Day 1–2</span>
          <span style={style.pill(COLORS.purple, COLORS.purpleDim)}>◈ Orient: Day 2–3</span>
          <span style={style.pill(COLORS.orange, COLORS.orangeDim)}>◆ Decide: Day 3–4</span>
          <span style={style.pill(COLORS.green, COLORS.greenDim)}>▶ Act: Day 4–7</span>
        </div>
      </div>

      {phases.map((phase) => (
        <div key={phase.id} style={{ marginBottom: "32px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "14px", marginBottom: "16px" }}>
            <div style={{
              width: "44px", height: "44px", background: phase.bg,
              border: `1px solid ${phase.color}`, borderRadius: "4px",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "20px", color: phase.color,
            }}>{phase.icon}</div>
            <div>
              <div style={{ fontSize: "18px", fontWeight: "800", color: phase.color, letterSpacing: "2px" }}>{phase.id}</div>
              <div style={{ fontSize: "11px", color: COLORS.muted }}>{phase.subtitle}</div>
            </div>
            <span style={{ marginLeft: "auto", ...style.badge(phase.color) }}>{phase.timeframe}</span>
          </div>

          <div style={{ borderLeft: `2px solid ${phase.color}`, paddingLeft: "20px" }}>
            {phase.items.map((item, i) => (
              <div key={i} style={{ marginBottom: "14px" }}>
                <div style={{ fontSize: "11px", color: phase.color, fontWeight: "700", letterSpacing: "1px", marginBottom: "4px" }}>
                  {String(i + 1).padStart(2, "0")} / {item.label}
                </div>
                <div style={{ fontSize: "12px", color: "#9898b8", lineHeight: "1.7" }}>{item.text}</div>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div style={style.card(COLORS.accentBorder)}>
        <div style={style.h3(COLORS.accent)}>⚡ OODA Loop is Continuous — Not One-Pass</div>
        <p style={{ ...style.p, marginBottom: 0 }}>
          After each Act sprint, loop back to Observe. Did demo work? Did threshold produce false positives? Re-orient, re-decide, re-act.
          The team that runs the fastest OODA loop wins — not the team with the best initial plan.
        </p>
      </div>
    </div>
  );
}

// ─── BUSINESS WORKFLOW TAB ──────────────────────────────────────────────────
function BusinessTab() {
  return (
    <div>
      <div style={{ marginBottom: "32px" }}>
        <div style={style.h2}>Business-Level Workflow</div>
        <p style={style.p}>End-to-end operational flow from the perspective of a Sports Organization using DAPS. Every step maps to a user action, a business outcome, and a success metric.</p>
      </div>

      {/* Phase 1: Onboarding */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.blue }}>Phase 01</span>
          <span>Asset Registration & Ownership Establishment</span>
          <div style={style.divLine} />
        </div>
        <div style={style.grid2}>
          <div style={style.card(COLORS.blueDim.replace("0.12", "0.4"))}>
            <div style={style.h3(COLORS.blue)}>User Journey</div>
            {["Sports club uploads original match photo / graphic", "Enters metadata: event, date, rights owner, license type", "System generates fingerprint + timestamps ownership record", "Receives registration certificate (asset ID + SHA256 hash)", "Asset is now tracked — any reuse anywhere is detectable"].map((t, i) => (
              <div key={i} style={style.step()}>
                <div style={style.stepNum(COLORS.blue)}>{i + 1}</div>
                <div style={{ fontSize: "12px", color: "#9898b8", lineHeight: "1.6" }}>{t}</div>
              </div>
            ))}
          </div>
          <div>
            <div style={style.card(COLORS.border)}>
              <div style={style.h3(COLORS.muted)}>Business Outcomes</div>
              <div style={{ fontSize: "12px", color: "#9898b8", lineHeight: "1.8" }}>
                ✦ Legal proof of ownership pre-violation<br />
                ✦ Chain of custody for DMCA / court use<br />
                ✦ Immutable timestamp prevents disputed ownership<br />
                ✦ SHA-256 hash is cryptographically admissible
              </div>
            </div>
            <div style={style.card(COLORS.border)}>
              <div style={style.h3(COLORS.muted)}>Success Metrics</div>
              <div style={{ fontSize: "12px", color: "#9898b8", lineHeight: "1.8" }}>
                ✦ Registration time &lt; 10 seconds per asset<br />
                ✦ 100% of uploaded assets fingerprinted<br />
                ✦ Zero duplicate registrations in index
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Phase 2: Detection */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.orange }}>Phase 02</span>
          <span>Suspected Infringement Detection</span>
          <div style={style.divLine} />
        </div>
        <div style={style.grid2}>
          <div style={style.card()}>
            <div style={style.h3(COLORS.orange)}>User Journey</div>
            {[
              "Rights team spots suspicious image on social / website",
              "Uploads query image to DAPS detection panel",
              "System runs 2-stage hybrid fingerprint check",
              "Returns: Similarity % + Classification + Modification Type",
              "Full evidence packet generated (JSON + visual report)",
            ].map((t, i) => (
              <div key={i} style={style.step()}>
                <div style={style.stepNum(COLORS.orange)}>{i + 1}</div>
                <div style={{ fontSize: "12px", color: "#9898b8", lineHeight: "1.6" }}>{t}</div>
              </div>
            ))}
          </div>
          <div>
            <div style={style.card(COLORS.border)}>
              <div style={style.h3(COLORS.orange)}>Detection Output (What User Sees)</div>
              <div style={style.codeBlock}>{`{
  "similarity_score": 0.84,
  "classification": "Modified Reuse",
  "confidence": 0.91,
  "modification_hints": [
    "crop_detected",
    "overlay_present",
    "color_shift"
  ],
  "matched_asset_id": "ASSET_2024_0031",
  "matched_owner": "FC Example Sports",
  "registered_on": "2024-11-10",
  "processing_time_ms": 142
}`}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Phase 3: Classification Decision */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.accent }}>Phase 03</span>
          <span>Classification & Business Decision Tree</span>
          <div style={style.divLine} />
        </div>
        <table style={style.table}>
          <thead>
            <tr>
              <th style={style.th}>Similarity</th>
              <th style={style.th}>System Label</th>
              <th style={style.th}>Business Meaning</th>
              <th style={style.th}>Recommended Action</th>
              <th style={style.th}>Urgency</th>
            </tr>
          </thead>
          <tbody>
            {[
              [">90%", "Original / Exact Match", "Identical copy redistributed without license", "Immediate DMCA / takedown notice", "🔴 Critical"],
              ["75–90%", "Modified Reuse", "Content edited but origin traceable — likely evading detection", "Legal review + evidence packet to counsel", "🟠 High"],
              ["60–75%", "Heavy Modification", "Composite or heavily edited — possible IP theft via manipulation", "Human review + flag for monitoring", "🟡 Medium"],
              ["<60%", "Unauthorized / Different", "Different content, no match — possible false report", "Log and dismiss, no action required", "🟢 Low"],
            ].map(([score, label, meaning, action, urgency], i) => (
              <tr key={i}>
                <td style={{ ...style.td, fontWeight: "700", color: [COLORS.red, COLORS.orange, COLORS.accent, COLORS.green][i] }}>{score}</td>
                <td style={{ ...style.td, fontWeight: "600", color: COLORS.text }}>{label}</td>
                <td style={style.td}>{meaning}</td>
                <td style={style.td}>{action}</td>
                <td style={{ ...style.td, fontWeight: "700" }}>{urgency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Phase 4: Reporting */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.purple }}>Phase 04</span>
          <span>Evidence Reporting & Legal Action</span>
          <div style={style.divLine} />
        </div>
        <div style={style.grid3}>
          {[
            { title: "Evidence Packet", color: COLORS.purple, points: ["Similarity score + confidence", "Modification type breakdown", "Original asset ID + owner", "Detection timestamp + hash", "Suitable for DMCA filing"] },
            { title: "Monitoring Dashboard", color: COLORS.blue, points: ["All registered assets visible", "Detection history per asset", "Trend: violations over time", "Export: CSV / PDF report", "Filter by severity / date"] },
            { title: "Business Intelligence", color: COLORS.orange, points: ["Which assets are targeted most", "Which platforms redistribute", "Modification patterns over time", "ROI: violations prevented", "League / club benchmark"] },
          ].map((card, i) => (
            <div key={i} style={style.card()}>
              <div style={style.h3(card.color)}>{card.title}</div>
              {card.points.map((p, j) => (
                <div key={j} style={{ fontSize: "12px", color: "#9898b8", marginBottom: "8px", display: "flex", gap: "8px" }}>
                  <span style={{ color: card.color }}>→</span>{p}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Business Value Summary */}
      <div style={{ ...style.card(COLORS.accentBorder), background: COLORS.accentDim }}>
        <div style={style.h3(COLORS.accent)}>💼 Business Value Proposition</div>
        <div style={style.grid2}>
          <div>
            <div style={{ fontSize: "12px", color: "#9898b8", lineHeight: "1.8" }}>
              <b style={{ color: COLORS.text }}>For Small/Mid-Tier Sports Orgs:</b><br />
              First accessible AI-powered IP protection tool. No enterprise contract required. Register → detect → act in under 5 minutes.
            </div>
          </div>
          <div>
            <div style={{ fontSize: "12px", color: "#9898b8", lineHeight: "1.8" }}>
              <b style={{ color: COLORS.text }}>Competitive Moat:</b><br />
              Modification type classification is the only feature that enables legal teams to determine intent of infringement — not just existence.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── TECHNICAL WORKFLOW TAB ─────────────────────────────────────────────────
function TechnicalTab() {
  return (
    <div>
      <div style={{ marginBottom: "32px" }}>
        <div style={style.h2}>Pure Technical Workflow</div>
        <p style={style.p}>System-level implementation blueprint. Every component, interface, data structure, and algorithm specified precisely.</p>
      </div>

      {/* Stage 1: Ingestion */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.blue }}>Stage 1</span>
          <span>Asset Ingestion & Preprocessing</span>
          <div style={style.divLine} />
        </div>
        <div style={style.grid2}>
          <div>
            {[
              { box: "POST /register", color: COLORS.blue, bg: COLORS.blueDim, sub: "FastAPI endpoint, multipart/form-data" },
              { box: "Input Validation", color: COLORS.muted, bg: COLORS.border + "44", sub: "MIME check, max 10MB, JPEG/PNG only" },
              { box: "PIL → RGB Normalize", color: COLORS.muted, bg: COLORS.border + "44", sub: "Convert all formats to consistent RGB tensor" },
              { box: "SHA-256 Hash", color: COLORS.orange, bg: COLORS.orangeDim, sub: "Immutable identity proof for provenance chain" },
              { box: "Dual-Path Embedding", color: COLORS.accent, bg: COLORS.accentDim, sub: "Splits into pHash path + SSCD path simultaneously" },
            ].map((item, i) => (
              <div key={i}>
                <div style={style.flowBox(item.color, item.bg)}>
                  {item.box}
                  <div style={{ fontSize: "10px", fontWeight: "400", color: COLORS.muted, marginTop: "4px" }}>{item.sub}</div>
                </div>
                {i < 4 && <div style={style.arrow}>↓</div>}
              </div>
            ))}
          </div>
          <div>
            <div style={style.card()}>
              <div style={style.h3(COLORS.blue)}>pHash Computation</div>
              <div style={style.codeBlock}>{`import imagehash
from PIL import Image

def compute_phash(img: Image) -> str:
    # Resize to 32x32, grayscale, DCT
    return str(imagehash.phash(img, hash_size=16))
    # Returns 256-bit perceptual hash
    # Hamming distance < 15 = likely copy`}</div>
            </div>
            <div style={style.card()}>
              <div style={style.h3(COLORS.purple)}>SSCD Embedding</div>
              <div style={style.codeBlock}>{`import torch
from torchvision import transforms
# Load facebookresearch/sscd-copy-detection
# Model: sscd_disc_mixup.torchscript.pt

preprocess = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
# Output: L2-normalized 512-dim vector`}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Stage 2: Storage */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.purple }}>Stage 2</span>
          <span>Vector Storage & Index Architecture</span>
          <div style={style.divLine} />
        </div>
        <div style={style.grid2}>
          <div style={style.card()}>
            <div style={style.h3(COLORS.purple)}>FAISS Index (Embedding Store)</div>
            <div style={style.codeBlock}>{`import faiss
import numpy as np

# IndexFlatIP = inner product
# (equivalent to cosine on L2-normed vectors)
DIM = 512  # SSCD output dimension
index = faiss.IndexFlatIP(DIM)

# With IDs for retrieval
index = faiss.IndexIDMap(
    faiss.IndexFlatIP(DIM)
)

def add_to_index(embedding: np.ndarray, 
                 asset_id: int):
    vec = embedding.reshape(1, -1)
    index.add_with_ids(vec, 
                       np.array([asset_id]))`}</div>
          </div>
          <div>
            <div style={style.card()}>
              <div style={style.h3(COLORS.orange)}>Metadata Store (SQLite Schema)</div>
              <div style={style.codeBlock}>{`CREATE TABLE assets (
  id         INTEGER PRIMARY KEY,
  filename   TEXT NOT NULL,
  sha256     TEXT UNIQUE NOT NULL,
  phash      TEXT NOT NULL,
  owner      TEXT NOT NULL,
  event_name TEXT,
  license    TEXT,
  registered_at DATETIME DEFAULT NOW,
  faiss_id   INTEGER UNIQUE
);

CREATE TABLE detections (
  id          INTEGER PRIMARY KEY,
  query_hash  TEXT,
  matched_id  INTEGER REFERENCES assets(id),
  sim_score   REAL,
  label       TEXT,
  mod_hints   TEXT,  -- JSON array
  detected_at DATETIME DEFAULT NOW
);`}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Stage 3: Detection Engine */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.orange }}>Stage 3</span>
          <span>2-Stage Hybrid Detection Engine</span>
          <div style={style.divLine} />
        </div>
        <div style={style.card(COLORS.orangeDim.replace("0.12", "0.4"))}>
          <div style={style.h3(COLORS.orange)}>Stage 3A — pHash Fast Filter (Gate 1)</div>
          <div style={style.codeBlock}>{`def phash_gate(query_phash: str, 
               candidate_phashes: list[str],
               threshold: int = 20) -> list[int]:
    """
    Hamming distance filter.
    Returns indices of candidates within threshold.
    Speed: O(n) but each comparison is ~1 microsecond.
    Eliminates 95%+ of obviously unrelated assets.
    """
    query_hash = imagehash.hex_to_hash(query_phash)
    matches = []
    for i, ph in enumerate(candidate_phashes):
        candidate = imagehash.hex_to_hash(ph)
        if (query_hash - candidate) <= threshold:
            matches.append(i)
    return matches
    # If matches empty → skip SSCD, return "Unauthorized"`}</div>
        </div>

        <div style={style.card()}>
          <div style={style.h3(COLORS.accent)}>Stage 3B — SSCD Semantic Search (Gate 2)</div>
          <div style={style.codeBlock}>{`def sscd_search(query_embedding: np.ndarray,
                faiss_index,
                k: int = 5) -> tuple[list[float], list[int]]:
    """
    FAISS inner product search on L2-normalized vectors
    = cosine similarity search.
    Returns top-k (scores, asset_ids).
    """
    vec = query_embedding.reshape(1, -1).astype('float32')
    D, I = faiss_index.search(vec, k)
    # D = similarity scores [0, 1]
    # I = matched asset IDs
    return D[0].tolist(), I[0].tolist()


def hybrid_score(phash_hamming: int,
                 cosine_sim: float) -> float:
    """
    Weighted fusion: pHash + SSCD.
    pHash normalized: 1 - (hamming / 256)
    SSCD cosine: already in [0, 1]
    """
    phash_score = 1.0 - (phash_hamming / 256.0)
    return 0.30 * phash_score + 0.70 * cosine_sim`}</div>
        </div>
      </div>

      {/* Stage 4: Classification */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.accent }}>Stage 4</span>
          <span>Multi-Class Classification + Modification Fingerprinting</span>
          <div style={style.divLine} />
        </div>
        <div style={style.grid2}>
          <div style={style.card()}>
            <div style={style.h3(COLORS.accent)}>Threshold Classifier</div>
            <div style={style.codeBlock}>{`def classify(hybrid_score: float) -> dict:
    if hybrid_score >= 0.90:
        return {
            "label": "Original",
            "severity": "critical",
            "action": "immediate_takedown"
        }
    elif hybrid_score >= 0.75:
        return {
            "label": "Modified_Reuse",
            "severity": "high",
            "action": "legal_review"
        }
    elif hybrid_score >= 0.60:
        return {
            "label": "Heavy_Modification",
            "severity": "medium",
            "action": "human_review"
        }
    else:
        return {
            "label": "Unauthorized",
            "severity": "low",
            "action": "dismiss"
        }`}</div>
          </div>
          <div style={style.card()}>
            <div style={style.h3(COLORS.red)}>Modification Hint Detector</div>
            <div style={style.codeBlock}>{`import cv2
import numpy as np

def detect_modifications(
    original: np.ndarray,
    query: np.ndarray
) -> list[str]:
    hints = []
    
    oh, ow = original.shape[:2]
    qh, qw = query.shape[:2]
    
    # Crop detection
    aspect_orig = ow / oh
    aspect_query = qw / qh
    if abs(aspect_orig - aspect_query) > 0.15:
        hints.append("crop_detected")
    
    # Color shift (mean channel diff)
    diff = np.abs(
        original.mean(axis=(0,1)) - 
        query.mean(axis=(0,1))
    )
    if diff.mean() > 15:
        hints.append("color_shift")
    
    # Overlay detection (edge density)
    e_orig = cv2.Canny(original, 50, 150)
    e_query = cv2.Canny(query, 50, 150)
    density_diff = (e_query.mean() - 
                    e_orig.mean())
    if density_diff > 8:
        hints.append("overlay_present")
    
    # Resolution change
    if abs(oh*ow - qh*qw) / (oh*ow) > 0.20:
        hints.append("resize_detected")
    
    return hints`}</div>
          </div>
        </div>
      </div>

      {/* Stage 5: API */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.green }}>Stage 5</span>
          <span>FastAPI Layer — Endpoint Specifications</span>
          <div style={style.divLine} />
        </div>
        <div style={style.card()}>
          <div style={style.codeBlock}>{`# POST /register
# Input:  multipart/form-data {file, owner, event_name, license}
# Output: {asset_id, sha256, phash, registered_at, message}
# SLA:    < 800ms

# POST /detect
# Input:  multipart/form-data {file}
# Output: EvidencePacket (full schema below)
# SLA:    < 500ms (target), < 1500ms (worst case, large index)

# GET /report/{detection_id}
# Input:  detection_id: str (path param)
# Output: Full EvidencePacket + matched asset metadata
# SLA:    < 100ms (DB read only)

# GET /assets
# Input:  owner: str (query param, optional)
# Output: List of registered assets with metadata
# SLA:    < 200ms

class EvidencePacket(BaseModel):
    detection_id:       str
    similarity_score:   float          # hybrid score [0,1]
    classification:     str            # Original/Modified_Reuse/Heavy_Modification/Unauthorized
    confidence:         float          # classification confidence
    modification_hints: list[str]      # crop/color_shift/overlay/resize
    matched_asset_id:   Optional[str]  # None if Unauthorized
    matched_owner:      Optional[str]
    registered_on:      Optional[str]
    processing_time_ms: int
    sha256_query:       str            # for provenance`}</div>
        </div>
      </div>

      {/* Stage 6: Eval */}
      <div style={{ marginBottom: "36px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.red }}>Stage 6</span>
          <span>Evaluation Protocol — Sports Augmentation Matrix</span>
          <div style={style.divLine} />
        </div>
        <table style={style.table}>
          <thead>
            <tr>
              <th style={style.th}>#</th>
              <th style={style.th}>Test Case</th>
              <th style={style.th}>Augmentation Applied</th>
              <th style={style.th}>Expected Score</th>
              <th style={style.th}>Expected Label</th>
              <th style={style.th}>Validates</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["T1", "Exact Copy", "None (same file)", ">95%", "Original", "Baseline correctness"],
              ["T2", "JPEG Re-compress", "Quality 60→30", ">90%", "Original", "Compression robustness"],
              ["T3", "10% Center Crop", "Crop 90% region", "78–88%", "Modified Reuse", "Crop detection"],
              ["T4", "Broadcast Overlay", "Channel logo pasted", "75–85%", "Modified Reuse", "Overlay hint"],
              ["T5", "Color Grade Shift", "Hue +30°, Sat +0.3", "72–83%", "Modified Reuse", "Color shift hint"],
              ["T6", "Instagram Reformat", "16:9 → 9:16 pad+crop", "70–82%", "Modified Reuse", "Resize + crop"],
              ["T7", "Heavy Composite", "50% opacity merge", "60–72%", "Heavy Modification", "Deep modification"],
              ["T8", "Different Image", "Completely unrelated", "<50%", "Unauthorized", "Specificity"],
            ].map(([id, name, aug, score, label, validates], i) => (
              <tr key={i}>
                <td style={{ ...style.td, color: COLORS.muted }}>{id}</td>
                <td style={{ ...style.td, fontWeight: "600", color: COLORS.text }}>{name}</td>
                <td style={style.td}>{aug}</td>
                <td style={{ ...style.td, fontWeight: "700", color: COLORS.accent }}>{score}</td>
                <td style={{ ...style.td, color: COLORS.orange }}>{label}</td>
                <td style={style.td}>{validates}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Stage 7: Stack */}
      <div style={{ marginBottom: "20px" }}>
        <div style={style.sectionTitle}>
          <span style={{ color: COLORS.muted }}>Stack</span>
          <span>Final Technology Decisions</span>
          <div style={style.divLine} />
        </div>
        <div style={style.grid3}>
          {[
            { layer: "Embedding", tech: "SSCD (Meta, MIT License)", why: "48% better than CLIP on DISC2021 copy detection benchmark", color: COLORS.purple },
            { layer: "Fast Filter", tech: "imagehash pHash (16-bit)", why: "Eliminates 95%+ of non-matches in microseconds, adversarial defense layer", color: COLORS.blue },
            { layer: "Vector DB", tech: "FAISS IndexIDMap + IndexFlatIP", why: "Sub-millisecond cosine search on L2-normalized vectors, scalable to millions", color: COLORS.orange },
            { layer: "Backend", tech: "FastAPI + Pydantic v2", why: "Async, typed, auto-generates OpenAPI docs, production ready", color: COLORS.green },
            { layer: "Metadata DB", tech: "SQLite → Postgres-ready schema", why: "Zero setup for hackathon, schema compatible for production migration", color: COLORS.accent },
            { layer: "Frontend", tech: "React + Tailwind CSS", why: "Looks production-grade to judges; Streamlit signals prototype", color: COLORS.red },
          ].map((item, i) => (
            <div key={i} style={style.card()}>
              <div style={{ fontSize: "10px", color: COLORS.muted, letterSpacing: "2px", textTransform: "uppercase", marginBottom: "6px" }}>{item.layer}</div>
              <div style={{ fontSize: "13px", fontWeight: "700", color: item.color, marginBottom: "8px" }}>{item.tech}</div>
              <div style={{ fontSize: "11px", color: "#6b6b8a", lineHeight: "1.6" }}>{item.why}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── MAIN ────────────────────────────────────────────────────────────────────
export default function DAPS() {
  const [tab, setTab] = useState("ooda");
  const tabs = [
    { id: "ooda", label: "OODA Loop" },
    { id: "biz", label: "Business Workflow" },
    { id: "tech", label: "Technical Workflow" },
  ];

  return (
    <div style={style.wrap}>
      <div style={style.header}>
        <div style={{
          width: "36px", height: "36px", background: COLORS.accentDim,
          border: `1px solid ${COLORS.accentBorder}`, borderRadius: "4px",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "16px",
        }}>⬡</div>
        <div>
          <div style={{ fontSize: "15px", fontWeight: "800", letterSpacing: "1px", color: COLORS.text }}>DAPS</div>
          <div style={{ fontSize: "10px", color: COLORS.muted, letterSpacing: "2px" }}>DIGITAL ASSET PROTECTION SYSTEM</div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: "8px" }}>
          <span style={style.tag("232,255,71")}>HACKATHON BUILD</span>
          <span style={style.tag("105,255,71")}>v1.0 PLAN</span>
        </div>
      </div>

      <div style={style.nav}>
        {tabs.map((t) => (
          <button key={t.id} style={style.navBtn(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={style.main}>
        {tab === "ooda" && <OodaTab />}
        {tab === "biz" && <BusinessTab />}
        {tab === "tech" && <TechnicalTab />}
      </div>
    </div>
  );
}
