# file: thoughtleadership_rewrite.py
# Purpose: Streamlit app to fetch Kobo submissions, score open-ended responses (0–3) using
#          exemplar fusion + rubric evidence, and export to Excel/Google Sheets.
#
# Key improvements vs thoughtleadership.py:
# - Uses a *distribution* over scores from multiple nearest exemplars (softmax KNN),
#   plus per-score centroids, rather than top-1 exemplar.
# - Replaces brittle word-overlap “off-topic” clamp with semantic question similarity.
# - Uses rubric evidence as a calibrated backstop (upgrade when exemplar confidence is low).
# - More robust JSONL reader (ignores blank lines, strips leading commas).
#
# NOTE: This file is designed to be drop-in for your Streamlit deployment.
#       Keep the same secrets keys you already have; new optional keys are documented below.
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional deps used in your original app
import gspread
from google.oauth2.service_account import Credentials
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer


# =============================================================================
# UI / STYLING
# =============================================================================
def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --primary: #F26A21;
            --primary-soft: #FDE7D6;
            --primary-soft-stronger: #FBD0AD;

            --gold: #FACC15;
            --gold-soft: #FEF9C3;
            --silver: #E5E7EB;

            --bg-main: #f5f5f5;
            --card-bg: #ffffff;
            --text-main: #111827;
            --text-muted: #6b7280;
            --border-subtle: #e5e7eb;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #FFF7ED 0, #F9FAFB 40%, #F3F4F6 100%);
            color: var(--text-main);
        }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #1f2937;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] * { color: #e5e7eb !important; }

        .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }

        h1, h2, h3 { font-family: "Segoe UI", system-ui, sans-serif; color: var(--text-main); }
        h1 { font-size: 2.1rem; font-weight: 700; }
        h2 { margin-top: 1.5rem; font-size: 1.3rem; }
        p, span, label { color: var(--text-muted); }

        .app-header-card {
            position: relative;
            background: radial-gradient(circle at top left, rgba(242,106,33,0.15), rgba(250,204,21,0.06), #ffffff);
            border-radius: 1.25rem;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.12);
            margin-bottom: 1.4rem;
            overflow: hidden;
        }
        .app-header-card::before {
            content: ""; position: absolute; inset: 0; height: 3px;
            background: linear-gradient(90deg, var(--gold-soft), var(--primary), var(--silver), var(--gold));
            opacity: 0.95;
        }
        .app-header-card::after {
            content: ""; position: absolute; bottom: -40px; right: -40px; width: 140px; height: 140px;
            background: radial-gradient(circle, rgba(250,204,21,0.35), transparent 60%); opacity: 0.7;
        }
        .app-header-subtitle { font-size: 0.9rem; color: var(--text-muted); }

        .pill {
            display: inline-block; font-size: 0.75rem; padding: 0.15rem 0.7rem; border-radius: 999px;
            background: rgba(242,106,33,0.08); border: 1px solid rgba(242,106,33,0.6);
            color: #9A3412; margin-bottom: 0.4rem;
        }

        .section-card {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border-subtle);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.04);
        }

        .stDataFrame table { font-size: 13px; border-radius: 0.75rem; overflow: hidden; border: 1px solid var(--border-subtle); }
        .stDataFrame table thead tr th { background-color: var(--primary-soft); font-weight: 600; color: #7c2d12; }

        .stDownloadButton button, .stButton button {
            border-radius: 999px !important;
            padding: 0.35rem 1.2rem !important;
            font-weight: 600 !important;
            border: 1px solid rgba(242,106,33,0.85) !important;
            background: linear-gradient(135deg, var(--primary) 0%, #FB923C 100%) !important;
            color: #FFFBEB !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
            box-shadow: 0 12px 25px rgba(248,113,22,0.45);
        }

        .stAlert { border-radius: 0.8rem; }
        #MainMenu, footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECRETS / PATHS
# =============================================================================
KOBO_BASE = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID1 = st.secrets.get("KOBO_ASSET_ID1", "")
KOBO_TOKEN = st.secrets.get("KOBO_TOKEN", "")
AUTO_PUSH = bool(st.secrets.get("AUTO_PUSH", False))

DATASETS_DIR = Path("DATASETS")
# Default deployment layout (same as your original Streamlit repo):
MAPPING_PATH = DATASETS_DIR / "mapping1.csv"
EXEMPLARS_PATH = DATASETS_DIR / "thought_leadership.cleaned.jsonl"

# Local/dev fallbacks (useful when running from a folder with uploaded files):
if not MAPPING_PATH.exists():
    alt = Path("/mnt/data/mapping1.csv")
    if alt.exists():
        MAPPING_PATH = alt
if not EXEMPLARS_PATH.exists():
    # prefer cleaned, else raw jsonl
    alt1 = Path("/mnt/data/thought_leadership.cleaned.jsonl")
    alt2 = Path("/mnt/data/thought_leadership.jsonl")
    if alt1.exists():
        EXEMPLARS_PATH = alt1
    elif alt2.exists():
        EXEMPLARS_PATH = alt2


# =============================================================================
# SCORING CONFIG (safe defaults)
# =============================================================================
BANDS = {0: "Counterproductive", 1: "Compliant", 2: "Strategic", 3: "Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 19, 21),
    ("Strategic Advisor", 14, 18),
    ("Emerging Advisor", 8, 13),
    ("Needs Capacity Support", 0, 7),
]

ORDERED_ATTRS = [
    "Locally Anchored Visioning",
    "Innovation and Insight",
    "Execution Planning",
    "Cross-Functional Collaboration",
    "Follow-Through Discipline",
    "Learning-Driven Adjustment",
    "Result-Oriented Decision-Making",
]

# Similarity search / fusion
KNN_K = int(st.secrets.get("KNN_K", 25))
KNN_TEMP = float(st.secrets.get("KNN_TEMP", 0.10))

# Sentence-level ensemble (prevents verbosity bias)
MAX_SENTS_USED = int(st.secrets.get("MAX_SENTS_USED", 8))        # cap evidence sentences per answer
SENT_MIN_WORDS = int(st.secrets.get("SENT_MIN_WORDS", 6))        # ignore very short sentences
MMR_CANDIDATES = int(st.secrets.get("MMR_CANDIDATES", 80))       # candidate pool before diversity selection
MMR_LAMBDA = float(st.secrets.get("MMR_LAMBDA", 0.80))           # 0..1, higher = prioritize relevance
CLUSTER_SIM = float(st.secrets.get("CLUSTER_SIM", 0.80))         # exemplar similarity threshold for "same meaning"
CONSENSUS_FLOOR = float(st.secrets.get("CONSENSUS_FLOOR", 0.45))

# Mixture weights across packs (question / attribute / global)
WQ_MIX = float(st.secrets.get("WQ_MIX", 0.70))
WA_MIX = float(st.secrets.get("WA_MIX", 0.20))
WG_MIX = float(st.secrets.get("WG_MIX", 0.10))
  # if consensus below, don't allow score=3

# Score fusion: how much weight to give KNN vs centroids
CENTROID_ALPHA = float(st.secrets.get("CENTROID_ALPHA", 0.60))  # 0..1

# Duplicate reuse
DUP_SIM = float(st.secrets.get("DUP_SIM", 0.94))

# Off-topic detection (semantic, not just word overlap)
OFFTOPIC_QSIM = float(st.secrets.get("OFFTOPIC_QSIM", 0.12))  # lower = less strict
OFFTOPIC_CAP = int(st.secrets.get("OFFTOPIC_CAP", 2))  # cap score to <= this if off-topic

# Lexical overlap only used as a weak signal (never alone to clamp)
MIN_QA_OVERLAP = float(st.secrets.get("MIN_QA_OVERLAP", 0.01))

# Rubric override controls
RUBRIC_OVERRIDE = bool(st.secrets.get("RUBRIC_OVERRIDE", True))
RUBRIC_UPGRADE_CONF = float(st.secrets.get("RUBRIC_UPGRADE_CONF", 0.65))
RUBRIC_DOWNGRADE = bool(st.secrets.get("RUBRIC_DOWNGRADE", False))
RUBRIC_SHOW_AUDIT = bool(st.secrets.get("RUBRIC_SHOW_AUDIT", False))

# Optional “confidence clamp” (rarely needed once fusion is used)
CONF_CLAMP = float(st.secrets.get("CONF_CLAMP", 0.0))  # 0 disables
CONF_CLAMP_TO = int(st.secrets.get("CONF_CLAMP_TO", 1))

# AI-likeness flagging (does not change score)
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

FUZZY_THRESHOLD = 80

PASSTHROUGH_HINTS = [
    "staff id", "staff_id", "staffid", "_id", "id", "_uuid", "uuid", "instanceid", "_submission_time",
    "submissiondate", "submission_date", "start", "_start", "end", "_end", "today", "date", "deviceid",
    "username", "enumerator", "submitted_via_web", "_xform_id_string", "formid", "assetid"
]
_EXCLUDE_SOURCE_COLS_LOWER = {
    "_id", "formhub/uuid", "start", "end", "today", "staff_id", "meta/instanceid",
    "_xform_id_string", "_uuid", "meta/rootuuid", "_submission_time", "_validation_status"
}

# =============================================================================
# TEXT HELPERS
# =============================================================================
_WORD_RX = re.compile(r"\w+")

def clean(s) -> str:
    """Normalize + trim; safe for NaN."""
    if s is None:
        return ""
    try:
        if isinstance(s, float) and s != s:
            return ""
    except Exception:
        pass
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s

def qa_overlap(ans: str, qtext: str) -> float:
    """Simple lexical overlap (weak signal only)."""
    ans_s = clean(ans).lower()
    q_s = clean(qtext).lower()
    at = set(re.findall(r"\w+", ans_s))
    qt = set(re.findall(r"\w+", q_s))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0


# =============================================================================
# ADAPTIVE STRICTNESS (AUTOMATIC)
# =============================================================================
def _adaptive_thresholds(ans: str, ex_conf: float, qsim: float, lex: float) -> dict:
    """
    Automatically adjust strictness per answer.
    Goals:
      - Avoid false low scores for long, detailed paraphrases (common in your dataset)
      - Still catch truly off-topic short answers
      - Be more cautious when exemplar confidence is low
    Returns a dict with:
      off_topic_qsim, min_lex, cap, cap_now_rule (callable)
    """
    a = clean(ans)
    n_chars = len(a)
    n_words = len(_WORD_RX.findall(a))

    # Base thresholds (lenient)
    off_qsim = OFFTOPIC_QSIM          # default 0.12 in lenient file
    min_lex  = MIN_QA_OVERLAP         # default 0.01 in lenient file
    cap      = OFFTOPIC_CAP           # default 2 in lenient file

    # If answer is long/detailed, relax topic thresholds (paraphrases)
    if n_chars >= 180 or n_words >= 35:
        off_qsim = min(off_qsim, 0.10)
        min_lex  = min(min_lex, 0.008)
        cap      = max(cap, 2)

    # If answer is very short, tighten topic checks (likely off-topic / low effort)
    if n_chars <= 70 or n_words <= 12:
        off_qsim = max(off_qsim, 0.14)
        min_lex  = max(min_lex, 0.015)
        cap      = min(cap, 1)  # short off-topic should not exceed 1

    # If exemplar confidence is low, be more willing to flag off-topic for review,
    # but don't automatically cap unless it's also short or extremely off-topic.
    low_conf = ex_conf is not None and ex_conf < 0.55

    # Extremely off-topic semantic signal
    extremely_off = (qsim is not None and qsim < (off_qsim - 0.05)) and (lex is not None and lex < (min_lex * 0.8))

    def cap_now(score: int) -> bool:
        if score is None:
            return False
        # Cap if:
        #  - answer is short, OR
        #  - extremely off-topic, OR
        #  - model is trying to give 3 while topic is clearly low
        if n_chars <= 120 or n_words <= 20:
            return True
        if extremely_off:
            return True
        if score >= 3 and qsim is not None and qsim < (off_qsim - 0.04):
            return True
        # If low confidence AND topic is low, cap only if score is high
        if low_conf and score >= 2 and qsim is not None and qsim < off_qsim:
            return True
        return False

    return {"off_qsim": off_qsim, "min_lex": min_lex, "cap": cap, "cap_now": cap_now}

# =============================================================================
# SENTENCE-LEVEL EXEMPLAR ENSEMBLE (KEY UPGRADE)
# =============================================================================
_SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+|\n+")

def split_sentences(text: str) -> List[str]:
    """
    Split into sentences/clauses robustly.
    We also split on newlines. Very short fragments are filtered later.
    """
    t = clean(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RX.split(t) if p and p.strip()]
    return parts

def _mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, sims: np.ndarray, k: int, lam: float) -> np.ndarray:
    """
    Maximal Marginal Relevance selection to diversify top exemplars.
    Works on a candidate set already sorted by similarity.
    """
    if doc_vecs.size == 0:
        return np.zeros((0,), dtype=np.int64)
    k = max(1, min(int(k), doc_vecs.shape[0]))
    lam = float(np.clip(lam, 0.0, 1.0))

    selected = []
    selected_mask = np.zeros(doc_vecs.shape[0], dtype=bool)

    # Precompute pairwise similarity on-demand against selected
    for _ in range(k):
        best_idx = None
        best_score = -1e9
        if not selected:
            # pick most similar first
            best_idx = int(np.argmax(sims))
            selected.append(best_idx)
            selected_mask[best_idx] = True
            continue

        sel_vecs = doc_vecs[selected]
        # diversity term: max similarity to any already selected exemplar
        div = np.max(doc_vecs @ sel_vecs.T, axis=1)  # (n,)
        mmr = lam * sims - (1.0 - lam) * div
        mmr[selected_mask] = -1e9
        best_idx = int(np.argmax(mmr))
        selected.append(best_idx)
        selected_mask[best_idx] = True

    return np.array(selected, dtype=np.int64)

def _consensus_score(doc_vecs: np.ndarray, labels: np.ndarray, sim_thr: float) -> Tuple[float, float, int]:
    """
    Estimate "same meaning" consensus among selected exemplars.
    We build a graph where edges connect exemplars with cosine sim >= sim_thr,
    then take the largest connected component.
    Returns: (consensus_share, purity, dominant_label)
    """
    n = doc_vecs.shape[0]
    if n <= 1:
        lab = int(labels[0]) if n == 1 else 0
        return 1.0 if n == 1 else 0.0, 1.0 if n == 1 else 0.0, lab

    sims = doc_vecs @ doc_vecs.T
    adj = sims >= float(sim_thr)
    np.fill_diagonal(adj, False)

    # union-find
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # connect edges
    xs, ys = np.where(adj)
    for a, b in zip(xs.tolist(), ys.tolist()):
        union(a, b)

    # component sizes
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    unique, counts = np.unique(roots, return_counts=True)
    largest_root = int(unique[int(np.argmax(counts))])
    comp_idx = np.where(roots == largest_root)[0]
    consensus_share = float(len(comp_idx) / n)

    comp_labels = labels[comp_idx]
    # purity in largest component
    vals, cts = np.unique(comp_labels, return_counts=True)
    dom_label = int(vals[int(np.argmax(cts))])
    purity = float(np.max(cts) / len(comp_idx)) if len(comp_idx) else 0.0
    return consensus_share, purity, dom_label

def score_text_against_pack(pack, ans: str, ans_vec: np.ndarray):
    """
    Sentence-level ensemble scoring:
      - Split answer into sentences (cap MAX_SENTS_USED)
      - For each sentence, retrieve diverse top-k exemplars (MMR) from a candidate pool
      - Convert exemplar similarities to a class distribution
      - Average distributions across sentences (prevents length bias)
      - Apply consensus penalty if exemplar meanings disagree
    """
    if pack is None or pack.vecs.size == 0 or ans_vec is None:
        return None

    sents = split_sentences(ans)
    if not sents:
        return None

    # Filter short sentences and cap count by selecting most informative (by word count)
    scored_sents = []
    for s in sents:
        w = len(_WORD_RX.findall(s))
        if w >= SENT_MIN_WORDS:
            scored_sents.append((w, s))
    if not scored_sents:
        # fallback to whole answer if nothing passes
        scored_sents = [(len(_WORD_RX.findall(ans)), ans)]

    scored_sents.sort(reverse=True, key=lambda x: x[0])
    picked = [s for _, s in scored_sents[:MAX_SENTS_USED]]

    embed_many(picked)
    sent_vecs = []
    for s in picked:
        v = emb_of(s)
        if v is not None:
            sent_vecs.append(v)
    if not sent_vecs:
        # fallback to whole answer vector
        sent_vecs = [ans_vec]

    dists = []
    max_sim_overall = -1.0
    margin_overall = 0.0
    consensus_penalties = []

    for sv in sent_vecs:
        sims_all = pack.vecs @ sv
        if sims_all.size == 0:
            continue

        # Candidate pool
        cN = max(10, min(int(MMR_CANDIDATES), sims_all.size))
        cand_idx = np.argpartition(-sims_all, cN - 1)[:cN]
        cand_idx = cand_idx[np.argsort(-sims_all[cand_idx])]

        cand_vecs = pack.vecs[cand_idx]
        cand_sims = sims_all[cand_idx]
        # Select diverse top-k
        k = max(1, min(int(KNN_K), cand_sims.size))
        sel_local = _mmr_select(sv, cand_vecs, cand_sims, k=k, lam=MMR_LAMBDA)
        sel_idx = cand_idx[sel_local]

        top_sims = sims_all[sel_idx]
        top_scores = pack.scores[sel_idx]

        # Softmax weights over similarities
        w = _softmax((top_sims - float(top_sims.max())) / float(KNN_TEMP))
        dist = np.zeros(4, dtype=np.float32)
        for s, wi in zip(top_scores, w):
            s = int(s)
            if 0 <= s <= 3:
                dist[s] += float(wi)

        # Centroid distribution as stabilizer
        cent_sims = pack.centroids @ sv
        cent_mask = (pack.counts >= 1).astype(np.float32)
        cent_sims = cent_sims * cent_mask + (-10.0) * (1.0 - cent_mask)
        cent_dist = _softmax(cent_sims / max(1e-6, float(KNN_TEMP)))

        alpha = float(CENTROID_ALPHA)
        if pack.vecs.shape[0] < 25:
            alpha = min(0.9, max(alpha, 0.75))

        dist = alpha * dist + (1.0 - alpha) * cent_dist
        dist = dist / (dist.sum() + 1e-9)

        # Consensus penalty: do selected exemplars agree on meaning?
        sel_vecs = pack.vecs[sel_idx]
        consensus_share, purity, dom_label = _consensus_score(sel_vecs, top_scores.astype(int), CLUSTER_SIM)
        # penalty is higher when agreement is low
        # Softer consensus penalty (avoid overly-strict confidence collapse)
        penalty = float(np.clip(1.0 - (0.75 * consensus_share + 0.25 * purity), 0.0, 0.35))
        consensus_penalties.append(penalty)

        dists.append(dist)

        if float(top_sims.max()) > max_sim_overall:
            max_sim_overall = float(top_sims.max())
            if len(top_sims) >= 2:
                margin_overall = float(np.sort(top_sims)[-1] - np.sort(top_sims)[-2])

    if not dists:
        return None

    # Aggregate across sentences WITHOUT length bonus:
    # We use a weighted average so strong, highly-supported sentences carry more weight,
    # but adding more sentences does not automatically increase the score.
    mat = np.vstack(dists)  # (S, 4)

    # Sentence weights: higher if distribution is confident and similarity is strong.
    # Use a mild floor so no single sentence dominates completely.
    sent_conf = np.max(mat, axis=1)  # (S,)
    # Convert max_sim proxies from penalties list length; we don't have per-sentence max_sim stored,
    # so use confidence as the primary weight.
    w = 0.15 + 0.85 * sent_conf
    w = w / (w.sum() + 1e-9)

    dist = (mat * w[:, None]).sum(axis=0)
    dist = dist / (dist.sum() + 1e-9)

    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())

    # Apply consensus penalty to confidence (not directly to score)
    if consensus_penalties:
        conf = float(np.clip(conf * (1.0 - float(np.mean(consensus_penalties))), 0.0, 1.0))

    # Prevent accidental 3s only when consensus/confidence is genuinely weak
    if pred >= 3 and conf < CONSENSUS_FLOOR:
        pred = 2

    return ScoreDist(dist=dist, expected=expected, pred=pred, conf=conf, max_sim=float(max_sim_overall), margin=float(margin_overall), method="sentence_mmr+centroid_weighted")

# =============================================================================
# AI DETECTION (your original logic, unchanged)
# =============================================================================
TRANSITION_OPEN_RX = re.compile(
    r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b",
    re.I,
)
LIST_CUES_RX = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX = re.compile(r"^[-*•]\s", re.M)
LONG_DASH_HARD_RX = re.compile(r"[—–]")
SYMBOL_RX = re.compile(
    r"[—–\-_]{2,}"
    r"|[≥≤≧≦≈±×÷%]"
    r"|[→←⇒↔↑↓]"
    r"|[•●◆▶✓✔✗❌§†‡]",
    re.U,
)
TIMEBOX_RX = re.compile(
    r"(?:\bday\s*\d+\b|\bweek\s*\d+\b|\bmonth\s*\d+\b|\bquarter\s*\d+\b"
    r"|\b\d+\s*(?:days?|weeks?|months?|quarters?)\b|\bby\s+day\s*\d+\b)",
    re.I,
)
AI_RX = re.compile(r"(?:as an ai\b|i am an ai\b)", re.I)
DAY_RANGE_RX = re.compile(r"\bday\s*\d+\s*[-–]\s*\d+\b", re.I)
PIPE_LIST_RX = re.compile(r"\s\|\s")
PARENS_ACRONYMS_RX = re.compile(r"\(([A-Z]{2,}(?:s)?(?:\s*,\s*[A-Z]{2,}(?:s)?)+).*?\)")
NUMBERED_BULLETS_RX = re.compile(r"\b\d+\s*[\.\)]\s*")
SLASH_PAIR_RX = re.compile(r"\b\w+/\w+\b")

AI_BUZZWORDS = {
    "minimum viable", "feedback loop", "trade-off", "evidence-based",
    "stakeholder alignment", "learners' agency", "learners’ agency",
    "norm shifts", "quick win", "low-lift", "scalable",
    "best practice", "pilot theatre", "timeboxed",
}

def ai_signal_score(text: str, question_hint: str = "") -> float:
    t = clean(text)
    if not t:
        return 0.0
    if LONG_DASH_HARD_RX.search(t):
        return 1.0

    score = 0.0
    if SYMBOL_RX.search(t): score += 0.35
    if TIMEBOX_RX.search(t): score += 0.15
    if AI_RX.search(t): score += 0.35
    if TRANSITION_OPEN_RX.search(t): score += 0.12
    if LIST_CUES_RX.search(t): score += 0.12
    if BULLET_RX.search(t): score += 0.08

    if DAY_RANGE_RX.search(t): score += 0.15
    if PIPE_LIST_RX.search(t): score += 0.10
    if PARENS_ACRONYMS_RX.search(t): score += 0.10
    if NUMBERED_BULLETS_RX.search(t): score += 0.12
    if SLASH_PAIR_RX.search(t): score += 0.08

    hits = 0
    for rx in (TIMEBOX_RX, DAY_RANGE_RX, PIPE_LIST_RX, NUMBERED_BULLETS_RX):
        if rx.search(t):
            hits += 1
    if hits >= 2: score += 0.25
    if hits >= 3: score += 0.15

    tl = t.lower()
    buzz_hits = sum(1 for b in AI_BUZZWORDS if b in tl)
    if buzz_hits:
        score += min(0.24, 0.08 * buzz_hits)

    if question_hint:
        overlap = qa_overlap(t, question_hint)
        if overlap < 0.06:
            score += 0.10

    return max(0.0, min(1.0, score))


# =============================================================================
# RUBRIC VALIDATOR (ported from your file; minor extensions)
# =============================================================================
def _has_any(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)

def _count_any(text: str, patterns: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for p in patterns if re.search(p, t))

def rubric_validate(qid: str, ans: str) -> Tuple[int, float, str]:
    """Return (rubric_score 0..3, rubric_conf 0..1, reason)."""
    a = clean(ans)
    if not a:
        return 0, 0.90, "empty"

    harmful = _has_any(a, [
        r"\bignore (the )?community\b",
        r"\bdrop (the )?gender\b",
        r"\bremove women\b",
        r"\bonly (work with )?big ngos?\b",
        r"\bjust scale fast\b",
        r"\bcentraliz(e|ing)\b.*\bcommunity\b",
    ])
    if harmful:
        return 0, 0.90, "harmful/derailing"

    if len(a) < 18 or len(_WORD_RX.findall(a)) < 4:
        return 1, 0.75, "too_short"

    qid = (qid or "").strip().upper()

    has_time = _has_any(a, [
        r"\b60 days\b", r"\b90 days\b", r"\bby friday\b", r"\bweekly\b", r"\bbi\-?weekly\b",
        r"\bquarterly\b", r"\b\d+\s*(days?|weeks?|months?)\b"
    ])
    has_owner = _has_any(a, [r"\bowner\b", r"\baccountable\b", r"\bresponsible\b", r"\blead\b", r"\bby name\b", r"\bRACI\b", r"\bescalat"])
    has_artifact = _has_any(a, [r"\bdashboard\b", r"\bdecision memo\b", r"\blearning log\b", r"\binsight brief\b", r"\bscorecard\b",
                                r"\bbacklog\b", r"\bagenda\b", r"\boutputs\b", r"\bchecklist\b", r"\bplaybook\b"])
    has_gate = _has_any(a, [r"\bdecision gate\b", r"\bgate\b", r"\bthreshold\b", r"\btrigger\b", r"\bif .* then\b"])
    has_tradeoff = _has_any(a, [r"\btrade\-?off\b", r"\bnon\-?negotiable\b", r"\bflex\b", r"\bguardrail\b", r"\bsafeguard\b", r"\bprotect\b"])
    has_local = _has_any(a, [r"\bgrassroots\b", r"\bcommunity voice\b", r"\blocal leadership\b", r"\bkujenga\b", r"\bparish\b", r"\bvsla(s)?\b", r"\bproducer groups?\b"])
    has_gender = _has_any(a, [r"\bwomen\b", r"\bgender\b", r"\bequity\b", r"\binclusion\b"])

    safeguards = _has_any(a, [
        r"\btor(s)?\b|\bterms of reference\b",
        r"\bbudget line\b|\bring\-?fence\b|\bprotected funding\b|\bpercentage\b.*\bbudget\b",
        r"\bgovernance\b|\bsteering committee\b|\bdecision rights?\b|\bco\-?own(ed)?\b",
        r"\bscorecard\b|\bgrievance\b|\baccountability\b|\bsigned changes\b",
    ])

    # ---- Per-question heuristics ----
    if qid == "LAV_Q1":
        # Broaden to allow “stakeholder engagement”, “SWOT”, “capacity needs” as core evidence of local anchoring
        core = _has_any(a, [
            r"women\-?led", r"producer groups?", r"local buying", r"buying days?",
            r"co\-?design", r"vsla(s)?", r"grassroots", r"community",
            r"stakeholder engagement", r"consult", r"swot", r"capacity needs?"
        ])
        flex = _has_any(a, [r"\bflex\b", r"which districts?", r"sequence", r"phased", r"onboard", r"aggregation", r"parish hubs?", r"delivery structure"])
        if core and flex and (safeguards or has_tradeoff):
            return 3, 0.80, "core+flex+safeguards"
        if core and flex:
            return 2, 0.78, "core+flex"
        if core or flex:
            return 1, 0.65, "partial(core/flex)"
        return 1, 0.55, "generic"

    if qid == "LAV_Q2":
        tor = _has_any(a, [r"\btor(s)?\b", r"terms of reference"])
        bud = _has_any(a, [r"\bbudget\b", r"\bbudget line\b", r"ring\-?fence", r"protected funding"])
        local = _has_any(a, [r"local leadership", r"community", r"women\-?led", r"kujenga", r"vsla"])
        enforce = _has_any(a, [r"decision rights?", r"sign\-?off", r"accountability", r"scorecard", r"minimum percentage"])
        if tor and bud and local and enforce:
            return 3, 0.80, "tor+budget+local+enforcement"
        if tor and bud and local:
            return 2, 0.75, "tor+budget+local"
        if local and (tor or bud):
            return 1, 0.60, "mentions local + (tor or budget)"
        return 1, 0.55, "generic"

    if qid == "LAV_Q3":
        risks = _has_any(a, [r"risk", r"sidelin(e|ed)", r"capture", r"elite", r"dilut", r"token", r"voice.*lost"])
        formal = _has_any(a, [r"formal partners?", r"big ngos?", r"large ngo", r"prime partner"])
        voice = _has_any(a, [r"community voice", r"local leadership", r"women", r"grassroots", r"kujenga"])
        if (formal or risks) and voice and safeguards:
            return 3, 0.78, "risks+voice+safeguards"
        if (formal or risks) and voice:
            return 2, 0.72, "risks+voice"
        if formal or risks:
            return 1, 0.60, "mentions risk/formal only"
        return 1, 0.55, "generic"

    if qid == "LAV_Q4":
        trade = _has_any(a, [r"trade\-?off", r"i would trade", r"drop", r"reduce", r"slow", r"phase"])
        protect = _has_any(a, [r"protect", r"non\-?negotiable", r"safeguard", r"guardrail"])
        why = _has_any(a, [r"because", r"so that", r"in order to"])
        if trade and protect and why and (has_local or has_gender):
            return 3, 0.75, "explicit trade-off + protection + reasoning"
        if trade and protect and why:
            return 2, 0.70, "trade-off + reasoning"
        if trade or protect:
            return 1, 0.60, "partial trade/protect"
        return 1, 0.55, "generic"

    # For the rest, reuse your evidence-rich fallback logic (keeps it stable)
    evidence_hits = sum([has_time, has_owner, has_artifact, has_gate, has_tradeoff, has_local, has_gender])
    if evidence_hits >= 4:
        return 2, 0.60, "fallback(evidence-rich)"
    if evidence_hits <= 1:
        return 1, 0.55, "fallback(generic)"
    return 1, 0.58, "fallback"


# =============================================================================
# EMBEDDINGS (batched, cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_embedder():
    # Keep same model to avoid surprises; can be swapped later.
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def _embed_texts_cached(texts_tuple: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    texts = list(texts_tuple)
    embs = get_embedder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {t: e for t, e in zip(texts, embs)}

_EMB_CACHE: Dict[str, np.ndarray] = {}

def embed_many(texts: List[str]) -> None:
    missing = [t for t in texts if t and t not in _EMB_CACHE]
    if not missing:
        return
    pack = _embed_texts_cached(tuple(missing))
    _EMB_CACHE.update(pack)

def emb_of(text: str) -> Optional[np.ndarray]:
    t = clean(text)
    return _EMB_CACHE.get(t, None)


# =============================================================================
# EXEMPLAR BANKS
# =============================================================================
@dataclass
class ScoreDist:
    dist: np.ndarray            # shape (4,)
    expected: float             # 0..3
    pred: int                   # 0..3
    conf: float                 # max(dist)
    max_sim: float              # max similarity in knn
    margin: float               # top1-top2
    method: str                 # "knn+centroid" etc.

@dataclass
class Any:
    vecs: np.ndarray            # (n, d)
    scores: np.ndarray          # (n,)
    texts: List[str]            # (n,)
    centroids: np.ndarray       # (4, d) normalized
    counts: np.ndarray          # (4,)


def _mix_dists(dists, weights):
    """
    Mix multiple ScoreDist distributions (question/attribute/global).
    Uses only the 'dist' field, and recomputes expected/pred/conf.
    """
    if dists is None or weights is None:
        return None
    present = [(d, float(w)) for d, w in zip(dists, weights) if d is not None and float(w) > 0]
    if not present:
        return None
    ws = np.array([w for _, w in present], dtype=np.float32)
    ws = ws / (ws.sum() + 1e-9)
    mat = np.vstack([d.dist for d, _ in present]).astype(np.float32)
    dist = (mat * ws[:, None]).sum(axis=0)
    dist = dist / (dist.sum() + 1e-9)
    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())
    max_sim = float(max(d.max_sim for d, _ in present))
    margin = float(max(d.margin for d, _ in present))
    method = "mix"
    return ScoreDist(dist=dist, expected=expected, pred=pred, conf=conf, max_sim=max_sim, margin=margin, method=method)

def _build_pack(texts: List[str], scores: List[int]) -> Any:
    # de-dup exact pairs
    seen = set()
    tt, ss = [], []
    for t, s in zip(texts, scores):
        t = clean(t)
        if not t:
            continue
        s = int(s) if str(s).isdigit() or isinstance(s, (int, np.integer)) else int(float(s))
        key = (t, s)
        if key in seen:
            continue
        seen.add(key)
        tt.append(t)
        ss.append(int(s))

    embed_many(list(set(tt)))

    vec_list, keep_t, keep_s = [], [], []
    for t, s in zip(tt, ss):
        v = emb_of(t)
        if v is None:
            continue
        vec_list.append(v)
        keep_t.append(t)
        keep_s.append(int(s))

    if vec_list:
        mat = np.vstack(vec_list).astype(np.float32)
    else:
        mat = np.zeros((0, 384), dtype=np.float32)

    scores_arr = np.array(keep_s, dtype=int)

    # per-score centroids
    centroids = np.zeros((4, mat.shape[1] if mat.size else 384), dtype=np.float32)
    counts = np.zeros(4, dtype=np.int32)
    for s in range(4):
        idx = np.where(scores_arr == s)[0]
        counts[s] = len(idx)
        if len(idx) >= 1:
            c = mat[idx].mean(axis=0)
            # normalize
            n = np.linalg.norm(c) + 1e-9
            centroids[s] = (c / n).astype(np.float32)
        else:
            # keep as zeros
            centroids[s] = np.zeros_like(centroids[s])

    return Any(vecs=mat, scores=scores_arr, texts=keep_t, centroids=centroids, counts=counts)

def build_exemplar_packs(exemplars: List[dict]):
    """Build packs by question_id and by attribute, plus global pack.
    Returns: (q_packs, a_packs, g_pack, by_qkey, question_texts)
    """
    by_qkey: Dict[str, dict] = {}
    by_attr: Dict[str, dict] = {}
    question_texts: List[str] = []
    all_texts: List[str] = []
    all_qtexts: List[str] = []

    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        qtext = clean(e.get("question_text", ""))
        txt = clean(e.get("text", ""))
        attr = clean(e.get("attribute", ""))
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0

        if not (qid or qtext) or not txt:
            continue

        key = qid if qid else qtext
        pack = by_qkey.setdefault(key, {"question_text": qtext, "scores": [], "texts": [], "attribute": attr})
        pack["scores"].append(sc)
        pack["texts"].append(txt)

        if qtext:
            question_texts.append(qtext)
            all_qtexts.append(qtext)

        if attr:
            by_attr.setdefault(attr, {"scores": [], "texts": []})
            by_attr[attr]["scores"].append(sc)
            by_attr[attr]["texts"].append(txt)

        all_texts.append(txt)

    # cache embeddings for all distinct exemplar texts and question texts (for semantic off-topic)
    embed_many(list(set(all_texts + all_qtexts)))

    q_packs = {k: _build_pack(v["texts"], v["scores"]) for k, v in by_qkey.items()}
    a_packs = {a: _build_pack(v["texts"], v["scores"]) for a, v in by_attr.items()}

    # global
    g_scores, g_texts = [], []
    for e in exemplars:
        t = clean(e.get("text", ""))
        if not t:
            continue
        try:
            s = int(e.get("score", 0))
        except Exception:
            s = 0
        g_texts.append(t)
        g_scores.append(s)
    g_pack = _build_pack(g_texts, g_scores)

    # de-dup question_texts
    seen = set()
    question_texts = [x for x in question_texts if not (x in seen or seen.add(x))]

    return q_packs, a_packs, g_pack, by_qkey, question_texts


def resolve_qkey(q_packs, by_qkey, question_texts, qid: str, prompt_hint: str) -> Optional[str]:
    qid = (qid or "").strip()
    if qid and qid in q_packs:
        return qid
    hint = clean(prompt_hint or "")
    if not (hint and question_texts):
        return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack.get("question_text", "")) == wanted:
                return k
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)

def score_vec_against_pack(pack: Any, vec: np.ndarray) -> Optional['ScoreDist']:
    """Return a fused score distribution from top-k exemplars + per-score centroids."""
    if pack is None or pack.vecs.size == 0 or vec is None:
        return None

    sims = pack.vecs @ vec  # cosine similarity (embeddings are normalized)
    if sims.size == 0:
        return None

    # --- KNN distribution ---
    k = max(1, min(int(KNN_K), sims.size))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    top_sims = sims[idx]
    top_scores = pack.scores[idx]

    # temperature softmax
    w = _softmax((top_sims - float(top_sims.max())) / float(KNN_TEMP))
    knn_dist = np.zeros(4, dtype=np.float32)
    for s, wi in zip(top_scores, w):
        s = int(s)
        if 0 <= s <= 3:
            knn_dist[s] += float(wi)

    # --- centroid distribution (more stable when there are many exemplars) ---
    cent_sims = pack.centroids @ vec  # (4,)
    # If a centroid is all zeros (no examples), push it down hard
    cent_mask = (pack.counts >= 1).astype(np.float32)
    cent_sims = cent_sims * cent_mask + (-10.0) * (1.0 - cent_mask)
    cent_dist = _softmax(cent_sims / max(1e-6, float(KNN_TEMP)))

    # --- fuse ---
    alpha = float(CENTROID_ALPHA)
    # if very few exemplars in any class, lean more on knn
    if pack.vecs.shape[0] < 25:
        alpha = min(0.9, max(alpha, 0.8))
    dist = alpha * knn_dist + (1.0 - alpha) * cent_dist
    dist = dist / (dist.sum() + 1e-9)

    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())
    max_sim = float(top_sims[0])
    margin = float(top_sims[0] - top_sims[1]) if len(top_sims) >= 2 else float(top_sims[0])
    return ScoreDist(dist=dist, expected=expected, pred=pred, conf=conf, max_sim=max_sim, margin=margin, method="knn+centroid")


# =============================================================================
# KOBO
# =============================================================================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID1 or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID1 and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions", "data"):
        url = kobo_url(KOBO_ASSET_ID1, kind)
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            payload = r.json()
            results = payload if isinstance(payload, list) else payload.get("results", [])
            if not results and "results" not in payload:
                results = payload
            df = pd.DataFrame(results)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
            return df
        except requests.HTTPError:
            if r.status_code in (401, 403):
                st.error("Kobo auth failed: check KOBO_TOKEN and tenant.")
                return pd.DataFrame()
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()
    st.error("Could not fetch Kobo data. Check KOBO_BASE/ASSET/TOKEN.")
    return pd.DataFrame()


# =============================================================================
# DATA LOADERS
# =============================================================================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Normalize expected columns: allow old naming
    if "prompt_hint" not in df.columns and "column" in df.columns:
        df = df.rename(columns={"column": "prompt_hint"})
    if "question_id" not in df.columns:
        raise ValueError("mapping.csv must include column 'question_id'")
    if "attribute" not in df.columns:
        raise ValueError("mapping.csv must include column 'attribute'")
    return df

def read_jsonl_path(path: Path) -> List[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # Fix common failure: a leading comma in a JSONL line
            if s.startswith(","):
                s = s.lstrip(",").strip()
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                # Attempt a last-resort fix: strip trailing commas
                s2 = re.sub(r",\s*$", "", s)
                out.append(json.loads(s2))
    return out


# =============================================================================
# KOBO COLUMN RESOLUTION (ported)
# =============================================================================
_QID_PREFIX_TO_SECTION = {
    "LAV": "A1",
    "II": "A2",
    "EP": "A3",
    "CFC": "A4",
    "FTD": "A5",
    "LDA": "A6",
    "RDM": "A7",
}

def _score_kobo_header(col: str, token: str) -> int:
    """Fuzzy score how well a Kobo header matches a token."""
    col = clean(col).lower()
    token = clean(token).lower()
    if not col or not token:
        return 0
    # reward direct containment and token_set match
    if token in col:
        return 100
    return int(fuzz.token_set_ratio(token, col))

def resolve_kobo_column_for_mapping(df_cols: List[str], qid: str, prompt_hint: str) -> Optional[str]:
    """Try best effort to locate the column containing a question's response."""
    qid = (qid or "").strip()
    if not qid:
        return None

    # direct match by question id or parts
    token = None
    if "_Q" in qid:
        try:
            prefix, qn = qid.split("_Q", 1)[0], int(qid.split("_Q")[-1])
        except Exception:
            prefix, qn = qid, None
        sect = _QID_PREFIX_TO_SECTION.get(prefix)
        if sect and qn is not None:
            token = f"{sect}_{qn}"

    if token:
        best, bs = None, 0
        for c in df_cols:
            sc = _score_kobo_header(c, token)
            if sc > bs:
                bs, best = sc, c
        if best and bs >= 82:
            return best

    hint = clean(prompt_hint or "")
    if hint:
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=5)
        for col, score, _ in hits:
            if score >= 80:
                return col

    return None


# =============================================================================
# SCORE COMBINATION LOGIC
# =============================================================================
def _combine_scores(
    ex: Optional['ScoreDist'],
    rub_sc: int,
    rub_conf: float,
    rub_reason: str,
) -> Tuple[int, int, float, str]:
    """Return (final_score, overrides_applied, final_conf, reason_tag)."""
    overrides = 0

    # Hard rubric for empty/harmful
    if rub_reason in ("empty", "harmful/derailing") and rub_conf >= 0.85:
        if ex is None:
            return int(rub_sc), 1, float(rub_conf), f"rubric:{rub_reason}"
        return int(min(ex.pred, rub_sc)), 1, float(rub_conf), f"rubric_cap:{rub_reason}"

    if ex is None:
        # fall back to rubric if reasonably confident
        if rub_conf >= 0.60:
            return int(rub_sc), 0, float(rub_conf), "rubric_only"
        return 1, 0, 0.35, "default_1"

    # Optional low-confidence clamp (rare; keep disabled by default)
    if CONF_CLAMP > 0 and ex.conf < CONF_CLAMP:
        if ex.pred > CONF_CLAMP_TO:
            overrides += 1
        ex_pred = min(int(ex.pred), int(CONF_CLAMP_TO))
        ex = ScoreDist(dist=ex.dist, expected=ex.expected, pred=ex_pred, conf=ex.conf, max_sim=ex.max_sim, margin=ex.margin, method=ex.method)

    if not RUBRIC_OVERRIDE:
        return int(ex.pred), 0, float(ex.conf), "exemplar_only"

    # If exemplar confidence is low and rubric is confident, lean to rubric (upgrade common).
    if ex.pred <= 1 and rub_sc >= 2 and rub_conf >= RUBRIC_UPGRADE_CONF and ex.conf < 0.88:
        overrides += 1
        return int(rub_sc), overrides, float(max(ex.conf, rub_conf)), "rubric_upgrade"

    # Optional downgrades for too short/generic answers (if enabled)
    if RUBRIC_DOWNGRADE and rub_sc <= 1 and rub_conf >= 0.85:
        # (we keep this conservative; real logic is in rubric_validate)
        if ex.pred > rub_sc:
            overrides += 1
            return int(rub_sc), overrides, float(max(ex.conf, rub_conf)), "rubric_downgrade"

    # Otherwise: blend expected score with rubric, weighted by confidence.
    # rubric weight increases as exemplar confidence decreases.
    w_rub = float(np.clip((rub_conf - 0.50) * (1.0 - ex.conf) * 1.6, 0.0, 0.55))
    w_ex = 1.0 - w_rub
    blended = w_ex * ex.expected + w_rub * float(rub_sc)
    pred = int(np.clip(int(round(blended)), 0, 3))
    return pred, overrides, float(max(ex.conf, rub_conf)), "blend"


# =============================================================================
# MAIN SCORER
# =============================================================================
def score_dataframe(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    q_packs: Dict[str, Any],
    a_packs: Dict[str, Any],
    g_pack: Any,
    by_qkey: Dict[str, dict],
    question_texts: List[str],
) -> pd.DataFrame:

    df_cols = list(df.columns)

    def want_col(c):
        lc = c.strip().lower()
        return any(h in lc for h in PASSTHROUGH_HINTS)

    passthrough_cols = [c for c in df_cols if want_col(c)]
    seen = set()
    passthrough_cols = [x for x in passthrough_cols if not (x in seen or seen.add(x))]

    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id", "staff_id", "staffid")), None)
    care_staff_col = next((c for c in df.columns if c.strip().lower() in ("care_staff", "care staff", "care-staff")), None)

    date_cols_pref = [
        "_submission_time", "SubmissionDate", "submissiondate",
        "end", "End", "start", "Start", "today", "date", "Date",
    ]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start"] if c in df.columns), None)
    end_col = next((c for c in ["end"] if c in df.columns), None)

    n_rows = len(df)

    # Date
    if date_col in df.columns:
        date_clean = df[date_col].astype(str).str.strip().str.lstrip(",")
        dt_series = pd.to_datetime(date_clean, errors="coerce")
    else:
        dt_series = pd.Series([pd.NaT] * n_rows)

    # duration
    if start_col:
        start_clean = df[start_col].astype(str).str.strip().str.lstrip(",")
        start_dt = pd.to_datetime(start_clean, utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)
    if end_col:
        end_clean = df[end_col].astype(str).str.strip().str.lstrip(",")
        end_dt = pd.to_datetime(end_clean, utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT] * n_rows)
    duration_min = (end_dt - start_dt).dt.total_seconds() / 60.0
    duration_min = duration_min.clip(lower=0)

    # Filter mapping
    all_mapping = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute", "")) in ORDERED_ATTRS]

    # Resolve Kobo columns per question id
    resolved_for_qid: Dict[str, str] = {}
    for r in all_mapping:
        qid = clean(r.get("question_id", ""))
        qhint = r.get("prompt_hint", "") or r.get("column", "")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit:
            resolved_for_qid[qid] = hit

    # Pre-embed all distinct answers and all question texts used for semantic off-topic
    distinct_answers: set = set()
    for _, row in df.iterrows():
        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a:
                    distinct_answers.add(a)
    # also embed question text / hints
    qtext_set = set()
    for r in all_mapping:
        qid = clean(r.get("question_id", ""))
        qhint = clean(r.get("prompt_hint", "") or r.get("column", ""))
        qkey = resolve_qkey(q_packs, by_qkey, question_texts, qid, qhint)
        if qkey:
            qt = clean((by_qkey.get(qkey, {}) or {}).get("question_text", ""))
            if qt:
                qtext_set.add(qt)
        if qhint:
            qtext_set.add(qhint)

    embed_many(list(distinct_answers | qtext_set))

    # Caches
    exact_sc_cache: Dict[Tuple[str, str], int] = {}
    dup_bank: Dict[str, List[Tuple[np.ndarray, int]]] = {}

    out_rows = []
    for i, resp in df.iterrows():
        row = {}
        row["Date"] = (
            pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(dt_series.iloc[i])
            else str(i)
        )
        val = duration_min.iloc[i]
        row["Duration"] = int(round(val)) if not pd.isna(val) else ""

        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(resp.get(who_col)) if who_col else ""

        # pass through selected metadata cols
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date", "Duration", "Care_Staff"):
                continue
            row[c] = resp.get(c, "")

        per_attr: Dict[str, List[int]] = {}
        any_ai = False
        override_count = 0
        needs_review = False

        # Pull answers once
        row_answers: Dict[str, str] = {}
        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                row_answers[qid] = clean(resp.get(col, ""))

        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            attr = clean(r.get("attribute", ""))
            qhint = clean(r.get("prompt_hint", "") or r.get("column", ""))
            col = resolved_for_qid.get(qid)
            if not col:
                continue

            ans = row_answers.get(qid, "")
            if not ans:
                continue

            vec = emb_of(ans)
            if vec is None:
                continue

            was_cached = (qid, ans) in exact_sc_cache
            cached_sc = exact_sc_cache.get((qid, ans))
            sc = cached_sc
            reused = False

            # duplicate reuse within question
            if sc is None:
                best_dup_sc, best_dup_sim = None, -1.0
                for v2, sc2 in dup_bank.get(qid, []):
                    sim = float(np.dot(vec, v2))
                    if sim > best_dup_sim:
                        best_dup_sim, best_dup_sc = sim, sc2
                if best_dup_sc is not None and best_dup_sim >= DUP_SIM:
                    sc = int(best_dup_sc)
                    reused = True

            # Exemplar scoring (question -> attribute -> global)
            ex: Optional['ScoreDist'] = None
            qkey = resolve_qkey(q_packs, by_qkey, question_texts, qid, qhint)
            qtext_full = ""
            if qkey:
                qtext_full = clean((by_qkey.get(qkey, {}) or {}).get("question_text", ""))

            if sc is None:
                # Score against ALL packs and mix (reduces strictness when question exemplars are sparse)
                ex_q = score_text_against_pack(q_packs[qkey], ans, vec) if (qkey and qkey in q_packs) else None
                ex_a = score_text_against_pack(a_packs[attr], ans, vec) if (attr in a_packs) else None
                ex_g = score_text_against_pack(g_pack, ans, vec)

                ex = _mix_dists([ex_q, ex_a, ex_g], [WQ_MIX, WA_MIX, WG_MIX])

            # Rubric
            rub_sc, rub_conf, rub_reason = rubric_validate(qid, ans)

            # Semantic off-topic check (conservative; marks review + caps only when clearly off-topic)
            off_topic = False
            qsim = None
            qref = qtext_full or qhint
            qv = emb_of(qref) if qref else None
            if qv is not None:
                qsim = float(np.dot(vec, qv))
                lex = qa_overlap(ans, qref)

                # Adaptive strictness per answer (automatic; no sidebar controls)
                ex_conf_for_adapt = ex.conf if ex is not None else 0.0
                thr = _adaptive_thresholds(ans, ex_conf_for_adapt, qsim, lex)

                # Off-topic flagging (semantic-first; lexical overlap is unreliable for paraphrases)
                if qsim < thr["off_qsim"]:
                    if len(ans) < 90:
                        if lex < thr["min_lex"]:
                            off_topic = True
                    else:
                        off_topic = True

            # Combine
            if sc is None:
                final_sc, overrides, final_conf, why = _combine_scores(ex, rub_sc, rub_conf, rub_reason)
                override_count += overrides
                sc = final_sc

                # Off-topic handling (after combining)
                # For open-ended questions, paraphrases can look "off-topic" lexically.
                # We therefore *mostly* flag for review and only cap when clearly unrelated and short.
                if off_topic:
                    needs_review = True
                    if (len(ans) < 90) and (qsim is not None and qsim < 0.06):
                        if sc > 1:
                            sc = 1

            # Cache scored result
            if sc is not None and not was_cached:
                exact_sc_cache[(qid, ans)] = int(sc)
                if not reused:
                    bank = dup_bank.setdefault(qid, [])
                    if len(bank) < 300:
                        bank.append((vec, int(sc)))

            # AI suspect flag
            ai_score = ai_signal_score(ans, qref)
            if ai_score >= AI_SUSPECT_THRESHOLD:
                any_ai = True

            # Determine Q number for output columns
            qn = None
            if "_Q" in (qid or ""):
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None

            if qn in (1, 2, 3, 4) and sc is not None:
                sk = f"{attr}_Qn{qn}"
                rk = f"{attr}_Rubric_Qn{qn}"
                row[sk] = int(sc)
                row[rk] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

            # Optional audit columns
            if RUBRIC_SHOW_AUDIT and qn in (1, 2, 3, 4):
                prefix = f"{attr}_Qn{qn}"
                if ex is not None:
                    row[f"{prefix}_ExPred"] = int(ex.pred)
                    row[f"{prefix}_ExExp"] = round(float(ex.expected), 3)
                    row[f"{prefix}_ExConf"] = round(float(ex.conf), 3)
                    row[f"{prefix}_ExMaxSim"] = round(float(ex.max_sim), 3)
                    row[f"{prefix}_ExDist"] = ",".join(f"{float(x):.3f}" for x in ex.dist.tolist())
                else:
                    row[f"{prefix}_ExPred"] = ""
                    row[f"{prefix}_ExExp"] = ""
                    row[f"{prefix}_ExConf"] = ""
                    row[f"{prefix}_ExMaxSim"] = ""
                    row[f"{prefix}_ExDist"] = ""
                row[f"{prefix}_RubricScore"] = int(rub_sc)
                row[f"{prefix}_RubricConf"] = round(float(rub_conf), 2)
                row[f"{prefix}_RubricReason"] = rub_reason
                row[f"{prefix}_QSim"] = "" if qsim is None else round(float(qsim), 3)
                row[f"{prefix}_OffTopic"] = bool(off_topic)

        # fill blanks
        for attr in ORDERED_ATTRS:
            for qn in (1, 2, 3, 4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + overall
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                row[f"{attr}_Avg (0–3)"] = ""
                row[f"{attr}_RANK"] = ""
            else:
                avg = float(np.mean(scores))
                band = int(round(avg))
                overall_total += band
                row[f"{attr}_Avg (0–3)"] = round(avg, 2)
                row[f"{attr}_RANK"] = BANDS[band]

        row["Overall Total (0–21)"] = overall_total
        row["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        row["Overrides_Applied"] = int(override_count)
        row["Needs_Review"] = bool(needs_review)
        row["AI_suspected"] = bool(any_ai)
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # column order similar to your app
    ordered = [c for c in ["Date", "Duration", "Care_Staff"] if c in res.columns]
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date", "Duration", "Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    mid_q = []
    for attr in ORDERED_ATTRS:
        for qn in (1, 2, 3, 4):
            mid_q += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
    ordered += [c for c in mid_q if c in res.columns]

    mid_a = []
    for attr in ORDERED_ATTRS:
        mid_a += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += [c for c in mid_a if c in res.columns]

    ordered += [c for c in ["Overall Total (0–21)", "Overall Rank"] if c in res.columns]
    for c in ["Overrides_Applied", "Needs_Review"]:
        if c in res.columns:
            ordered.append(c)
    if "AI_suspected" in res.columns:
        ordered.append("AI_suspected")

    # Append audit columns at the end (if enabled)
    if RUBRIC_SHOW_AUDIT:
        audit_cols = [c for c in res.columns if c not in ordered]
        ordered += audit_cols

    return res.reindex(columns=[c for c in ordered if c in res.columns])


# =============================================================================
# EXPORTS / SHEETS
# =============================================================================
def _ensure_ai_last(df: pd.DataFrame, export_name: str = "AI_Suspected", source_name: str = "AI_suspected") -> pd.DataFrame:
    out = df.copy()
    if export_name not in out.columns:
        if source_name in out.columns:
            out = out.rename(columns={source_name: export_name})
        else:
            out[export_name] = ""
    cols = [c for c in out.columns if c != export_name] + [export_name]
    return out[cols]

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    df_out = _ensure_ai_last(df)
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df_out.to_excel(w, index=False)
    return bio.getvalue()


# Google Sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME1", "Thought Leadership")

def _normalize_sa_dict(raw: dict) -> dict:
    if not raw:
        raise ValueError("gcp_service_account missing in secrets.")
    sa = dict(raw)
    if "token_ur" in sa and "token_uri" not in sa:
        sa["token_uri"] = sa.pop("token_ur")
    if sa.get("private_key") and "\\n" in sa["private_key"]:
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    sa.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    sa.setdefault("auth_uri", "https://accounts.google.com/o/oauth2/auth")
    sa.setdefault("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs")
    required = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id", "token_uri"]
    missing = [k for k in required if not sa.get(k)]
    if missing:
        raise ValueError(f"gcp_service_account missing fields: {', '.join(missing)}")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets.get("gcp_service_account"))
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

def _open_ws_by_key() -> gspread.Worksheet:
    key = st.secrets.get("GSHEETS_SPREADSHEET_KEY")
    ws_name = DEFAULT_WS_NAME
    if not key:
        raise ValueError("GSHEETS_SPREADSHEET_KEY not set in secrets.")
    gc = gs_client()
    sh = gc.open_by_key(key)
    try:
        return sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=ws_name, rows="20000", cols="150")

def _post_write_formatting(ws: gspread.Worksheet, cols: int) -> None:
    try:
        ws.freeze(rows=1)
    except Exception:
        pass
    try:
        ws.spreadsheet.batch_update(
            {
                "requests": [
                    {
                        "autoResizeDimensions": {
                            "dimensions": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": cols}
                        }
                    }
                ]
            }
        )
    except Exception:
        pass

def upload_df_to_gsheets(df: pd.DataFrame) -> Tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        df_out = _ensure_ai_last(df, export_name="AI_Suspected", source_name="AI_suspected")
        header = df_out.columns.astype(str).tolist()
        values = df_out.astype(object).where(pd.notna(df_out), "").values.tolist()
        ws.clear()
        ws.spreadsheet.values_batch_update(
            body={"valueInputOption": "USER_ENTERED", "data": [{"range": f"'{ws.title}'!A1", "values": [header] + values}]}
        )
        _post_write_formatting(ws, len(header))
        return True, f"✅ Wrote {len(values)} rows × {len(header)} cols to '{ws.title}' (last='AI_Suspected')."
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()

    st.markdown(
        """
        <div class="app-header-card">
            <div class="pill">Thought Leadership • Auto Scoring</div>
            <h1>Thought Leadership</h1>
            <p class="app-header-subtitle">
                Import Kobo submissions, score open-ended responses (exemplar fusion + rubric evidence),
                flag AI-like patterns, and export results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls (safe & transparent)
    st.sidebar.header("Scoring controls")
    

    try:
        mapping = load_mapping_from_path(MAPPING_PATH)
    except Exception as e:
        st.error(f"Failed to load mapping: {e}")
        return

    try:
        exemplars = read_jsonl_path(EXEMPLARS_PATH)
        if not exemplars:
            st.error("Exemplars file is empty.")
            return
    except Exception as e:
        st.error(f"Failed to read exemplars: {e}")
        return

    with st.spinner("Building exemplar banks (fusion scorer)..."):
        q_packs, a_packs, g_pack, by_q, qtexts = build_exemplar_packs(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Fetched dataset")
    st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns):,}")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("Scoring (+ AI flagging)..."):
        scored = score_dataframe(df, mapping, q_packs, a_packs, g_pack, by_q, qtexts)

    st.success("✅ Scoring complete.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Scored table")
    st.caption("Fusion scoring reduces ‘random 1s’ caused by nearest single exemplar or brittle off-topic checks.")
    styled = scored.style.apply(
        lambda r: ["background-color: #241E4E"] * len(r) if ("AI_suspected" in r and r["AI_suspected"]) else ["" for _ in r],
        axis=1,
    )
    st.dataframe(styled, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⬇️ Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Excel",
            data=to_excel_bytes(scored),
            file_name="Leadership_Scoring.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=_ensure_ai_last(scored).to_csv(index=False).encode("utf-8"),
            file_name="Leadership_Scoring.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if AUTO_PUSH:
        with st.spinner("📤 Sending to Google Sheets..."):
            ok, msg = upload_df_to_gsheets(scored)
        (st.success if ok else st.error)(msg)


if __name__ == "__main__":
    main()
