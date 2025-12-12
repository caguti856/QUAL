# file: thoughtleadership_centroids_fast.py
# FAST scoring: per-question centroids (0–3) + simple clamps (off-topic + low confidence)
# Kobo fetch + mapping resolution + exports to Excel/CSV + optional Google Sheets push

from __future__ import annotations

import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

# Optional Google Sheets push
import gspread
from google.oauth2.service_account import Credentials


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
            --gold: #FACC15;
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

        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1, h2, h3 { color: var(--text-main); }
        p, span, label { color: var(--text-muted); }

        .app-header-card {
            position: relative;
            background: radial-gradient(circle at top left,
                rgba(242,106,33,0.15),
                rgba(250,204,21,0.06),
                #ffffff);
            border-radius: 1.25rem;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.12);
            margin-bottom: 1.2rem;
            overflow: hidden;
        }
        .app-header-card::before {
            content: "";
            position: absolute;
            inset: 0;
            height: 3px;
            background: linear-gradient(90deg, #FEF9C3, var(--primary), var(--silver), var(--gold));
            opacity: 0.95;
        }

        .pill {
            display: inline-block;
            font-size: 0.75rem;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            background: rgba(242,106,33,0.08);
            border: 1px solid rgba(242,106,33,0.6);
            color: #9A3412;
            margin-bottom: 0.35rem;
        }

        .section-card {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border-subtle);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.04);
        }

        .stDownloadButton button, .stButton button {
            border-radius: 999px !important;
            padding: 0.35rem 1.2rem !important;
            font-weight: 600 !important;
            border: 1px solid rgba(242,106,33,0.85) !important;
            background: linear-gradient(135deg, var(--primary) 0%, #FB923C 100%) !important;
            color: #FFFBEB !important;
        }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
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
MAPPING_PATH = DATASETS_DIR / "mapping1.csv"
EXEMPLARS_PATH = DATASETS_DIR / "thought_leadership.cleaned.jsonl"

# local fallbacks (helpful in dev)
if not MAPPING_PATH.exists():
    alt = Path("/mnt/data/mapping1.csv")
    if alt.exists():
        MAPPING_PATH = alt
if not EXEMPLARS_PATH.exists():
    alt1 = Path("/mnt/data/thought_leadership.cleaned.jsonl")
    alt2 = Path("/mnt/data/thought_leadership.jsonl")
    if alt1.exists():
        EXEMPLARS_PATH = alt1
    elif alt2.exists():
        EXEMPLARS_PATH = alt2


# =============================================================================
# CONSTANTS
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

# clamps (meaning only, not length)
MIN_QA_OVERLAP = float(st.secrets.get("MIN_QA_OVERLAP", 0.05))  # off-topic clamp gate
CONF_CLAMP = float(st.secrets.get("CONF_CLAMP", 0.0))          # 0 disables; e.g. 0.45 clamps to <=1
SOFTMAX_TEMP = float(st.secrets.get("SOFTMAX_TEMP", 0.10))     # centroid softmax temperature

# For mapping resolution
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
# AI SUSPECT (your working logic)
# =============================================================================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

TRANSITION_OPEN_RX = re.compile(
    r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b",
    re.I
)
LIST_CUES_RX = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX = re.compile(r"^[-*•]\s", re.M)
LONG_DASH_HARD_RX = re.compile(r"[—–]")
SYMBOL_RX = re.compile(
    r"[—–\-_]{2,}"
    r"|[≥≤≧≦≈±×÷%]"
    r"|[→←⇒↔↑↓]"
    r"|[•●◆▶✓✔✗❌§†‡]",
    re.U
)
TIMEBOX_RX = re.compile(
    r"(?:\bday\s*\d+\b|\bweek\s*\d+\b|\bmonth\s*\d+\b|\bquarter\s*\d+\b"
    r"|\b\d+\s*(?:days?|weeks?|months?|quarters?)\b|\bby\s+day\s*\d+\b)",
    re.I
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
    "best practice", "pilot theatre", "timeboxed"
}


# =============================================================================
# TEXT HELPERS
# =============================================================================
_WORD_RX = re.compile(r"\w+")


def clean(s) -> str:
    if s is None:
        return ""
    try:
        if isinstance(s, float) and s != s:  # NaN
            return ""
    except Exception:
        pass
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def qa_overlap(ans, qtext) -> float:
    ans_s = clean(ans).lower()
    q_s = clean(qtext).lower()
    at = set(_WORD_RX.findall(ans_s))
    qt = set(_WORD_RX.findall(q_s))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0


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
# MAPPING + EXEMPLARS
# =============================================================================
QID_PREFIX_TO_SECTION = {"LAV": "A1", "II": "A2", "EP": "A3", "CFC": "A4", "FTD": "A5", "LDA": "A6", "RDM": "A7"}
QNUM_RX = re.compile(r"_Q(\d+)$")


def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")

    m = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    m.columns = [c.strip() for c in m.columns]

    # accept both "column" or "prompt_hint"
    if "prompt_hint" not in m.columns and "column" in m.columns:
        m = m.rename(columns={"column": "prompt_hint"})
    required = {"question_id", "attribute"}
    if not required.issubset(set(m.columns)):
        raise ValueError(f"mapping needs columns: {', '.join(sorted(required))} (+ optional prompt_hint)")

    # snap attribute to canonical labels
    norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
    target = {norm(a): a for a in ORDERED_ATTRS}

    def snap_attr(a):
        key = norm(a)
        if key in target:
            return target[key]
        best = process.extractOne(key, list(target.keys()), scorer=fuzz.token_set_ratio)
        return target[best[0]] if best and best[1] >= 75 else None

    m["attribute"] = m["attribute"].apply(snap_attr)
    m = m[m["attribute"].notna()].copy()
    if "prompt_hint" not in m.columns:
        m["prompt_hint"] = ""
    return m


def read_jsonl_path(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exemplars file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # tolerate accidental leading commas
            if s.startswith(","):
                s = s.lstrip(",").strip()
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                # tolerate trailing commas
                s2 = re.sub(r",\s*$", "", s)
                rows.append(json.loads(s2))
    return rows


def build_kobo_base_from_qid(question_id: str) -> Optional[List[str]]:
    if not question_id:
        return None
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m:
        return None
    qn = m.group(1)
    prefix = qid.split("_Q")[0]
    sect = QID_PREFIX_TO_SECTION.get(prefix)
    if not sect:
        return None
    token = f"{sect}_{qn}"
    roots = ["Thought Leadership", "Leadership"]
    return [f"{root}/{sect}_Section/{token}" for root in roots]


def expand_possible_kobo_columns(base: str) -> List[str]:
    if not base:
        return []
    return [
        base,
        f"{base} :: Answer (text)",
        f"{base} :: English (en)",
        f"{base} - English (en)",
        f"{base}_labels",
        f"{base}_label",
    ]


def _score_kobo_header(col: str, token: str) -> int:
    c = col.lower()
    t = token.lower()
    if c == t:
        return 100
    s = 0
    if c.endswith("/" + t): s = max(s, 95)
    if f"/{t}/" in c: s = max(s, 92)
    if f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c: s = max(s, 90)
    if t in c: s = max(s, 80)
    if "english" in c or "label" in c or "(en)" in c: s += 5
    if "answer (text)" in c or "answer_text" in c or "text" in c: s += 5
    if "thought leadership/" in c or "leadership/" in c or "/a" in c: s += 2
    return s


def resolve_kobo_column_for_mapping(df_cols: List[str], question_id: str, prompt_hint: str) -> Optional[str]:
    bases = build_kobo_base_from_qid(question_id) or []
    variants = []
    for base in bases:
        variants.extend(expand_possible_kobo_columns(base))

    # exact matches first
    for v in variants:
        if v in df_cols:
            return v

    # prefix match (KoBo sometimes adds suffixes)
    for c in df_cols:
        if any(c.startswith(b) for b in bases):
            return c

    # token match (A1_1, etc.)
    token = None
    if question_id:
        qid = question_id.strip().upper()
        m = QNUM_RX.search(qid)
        if m:
            qn = m.group(1)
            prefix = qid.split("_Q")[0]
            sect = QID_PREFIX_TO_SECTION.get(prefix)
            if sect:
                token = f"{sect}_{qn}"

    if token:
        best, bs = None, 0
        for c in df_cols:
            sc = _score_kobo_header(c, token)
            if sc > bs:
                bs, best = sc, c
        if best and bs >= 82:
            return best

    # prompt hint fallback
    hint = clean(prompt_hint or "")
    if hint:
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=5)
        for col, score, _ in hits:
            if score >= FUZZY_THRESHOLD:
                return col

    return None


# =============================================================================
# EMBEDDINGS (fast + cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    # Small + fast; meaning-based; avoids “length reward”
    return SentenceTransformer(st.secrets.get("EMBEDDER_NAME", "all-MiniLM-L6-v2"))


_EMB_CACHE: Dict[str, np.ndarray] = {}


def embed_many(texts: List[str]) -> None:
    texts = [clean(t) for t in texts if clean(t)]
    if not texts:
        return

    missing = [t for t in texts if t not in _EMB_CACHE]
    if not missing:
        return

    model = get_embedder()
    # chunk to avoid memory spikes
    bs = int(st.secrets.get("EMBED_BATCH", 64))
    for start in range(0, len(missing), bs):
        chunk = missing[start:start + bs]
        embs = model.encode(
            chunk,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=bs,
        ).astype(np.float32)
        for t, v in zip(chunk, embs):
            _EMB_CACHE[t] = v


def emb_of(text: str) -> Optional[np.ndarray]:
    t = clean(text)
    return _EMB_CACHE.get(t)


# =============================================================================
# PER-QUESTION CENTROIDS (THE FIX)
# =============================================================================
def build_question_centroids(exemplars: List[dict]):
    """
    Returns:
      q_centroids[qid]  -> {"centroids": (4,d), "counts": (4,)}
      a_centroids[attr] -> {"centroids": (4,d), "counts": (4,)}  fallback
      g_centroids       -> {"centroids": (4,d), "counts": (4,)}  fallback
    """
    by_q: Dict[str, Dict[str, list]] = {}
    by_a: Dict[str, Dict[str, list]] = {}
    g_texts: List[str] = []
    g_scores: List[int] = []

    # collect
    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        attr = clean(e.get("attribute", ""))
        txt = clean(e.get("text", ""))
        if not txt:
            continue
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0
        sc = max(0, min(3, sc))

        if qid:
            by_q.setdefault(qid, {"texts": [], "scores": []})
            by_q[qid]["texts"].append(txt)
            by_q[qid]["scores"].append(sc)

        if attr:
            by_a.setdefault(attr, {"texts": [], "scores": []})
            by_a[attr]["texts"].append(txt)
            by_a[attr]["scores"].append(sc)

        g_texts.append(txt)
        g_scores.append(sc)

    # embed all unique exemplar texts once
    embed_many(list(set(g_texts)))

    d = int(get_embedder().get_sentence_embedding_dimension())

    def _centroids_from(texts: List[str], scores: List[int]) -> dict:
        sums = np.zeros((4, d), dtype=np.float32)
        cnts = np.zeros(4, dtype=np.int32)

        for t, s in zip(texts, scores):
            v = emb_of(t)
            if v is None:
                continue
            sums[int(s)] += v
            cnts[int(s)] += 1

        cents = np.zeros_like(sums)
        for k in range(4):
            if cnts[k] > 0:
                cents[k] = sums[k] / float(cnts[k])

        # normalize each centroid for cosine sim
        norms = np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
        cents = cents / norms
        return {"centroids": cents.astype(np.float32), "counts": cnts.astype(np.int32)}

    q_centroids = {qid: _centroids_from(v["texts"], v["scores"]) for qid, v in by_q.items()}
    a_centroids = {a: _centroids_from(v["texts"], v["scores"]) for a, v in by_a.items()}
    g_centroids = _centroids_from(g_texts, g_scores)

    return q_centroids, a_centroids, g_centroids


def score_vec_against_centroids(pack: dict, vec: np.ndarray) -> Tuple[Optional[int], float]:
    """
    pack: {"centroids": (4,d), "counts": (4,)}
    vec: normalized embedding
    """
    if pack is None or vec is None:
        return None, 0.0

    cents = pack["centroids"]
    cnts = pack["counts"]
    valid = (cnts > 0)
    if not valid.any():
        return None, 0.0

    sims = cents @ vec  # (4,)
    sims = np.where(valid, sims, -1e9)

    # softmax over 4 classes
    x = sims - np.max(sims)
    w = np.exp(x / float(SOFTMAX_TEMP))
    w = np.where(valid, w, 0.0)
    w = w / (w.sum() + 1e-9)

    pred = int(np.argmax(w))
    conf = float(np.max(w))
    return pred, conf


# =============================================================================
# SCORING DATAFRAME
# =============================================================================
def score_dataframe(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    q_centroids: dict,
    a_centroids: dict,
    g_centroids: dict,
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

    date_cols_pref = ["_submission_time", "SubmissionDate", "submissiondate", "end", "End", "start", "Start", "today", "date", "Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start"] if c in df.columns), None)
    end_col = next((c for c in ["end"] if c in df.columns), None)

    n_rows = len(df)

    # Dates
    if date_col in df.columns:
        date_clean = df[date_col].astype(str).str.strip().str.lstrip(",")
        dt_series = pd.to_datetime(date_clean, errors="coerce")
    else:
        dt_series = pd.Series([pd.NaT] * n_rows)

    if start_col:
        start_dt = pd.to_datetime(df[start_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)

    if end_col:
        end_dt = pd.to_datetime(df[end_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT] * n_rows)

    duration_min = (end_dt - start_dt).dt.total_seconds() / 60.0
    duration_min = duration_min.clip(lower=0)

    # Keep only expected attributes
    all_mapping = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute", "")) in ORDERED_ATTRS]

    # Resolve Kobo columns for each question_id
    resolved_for_qid: Dict[str, str] = {}
    for r in all_mapping:
        qid = clean(r.get("question_id", ""))
        qhint = r.get("prompt_hint", "") or ""
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit:
            resolved_for_qid[qid] = hit

    # Embed all distinct answers once
    distinct_answers = set()
    for _, row in df.iterrows():
        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a:
                    distinct_answers.add(a)
    embed_many(list(distinct_answers))

    # cache exact repeated (qid, answer) to score
    exact_sc_cache: Dict[Tuple[str, str], Tuple[int, float, float]] = {}  # score, conf, ai_score

    out_rows = []
    for i, resp in df.iterrows():
        row = {}

        row["Date"] = (
            pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(dt_series.iloc[i]) else str(i)
        )
        val = duration_min.iloc[i]
        row["Duration"] = int(round(val)) if not pd.isna(val) else ""

        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(resp.get(who_col)) if who_col else ""

        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date", "Duration", "Care_Staff"):
                continue
            row[c] = resp.get(c, "")

        per_attr: Dict[str, List[int]] = {}
        any_ai = False

        # Score each mapped question
        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            attr = clean(r.get("attribute", ""))
            qhint = clean(r.get("prompt_hint", ""))

            col = resolved_for_qid.get(qid)
            if not col or col not in df.columns:
                continue

            ans = clean(resp.get(col, ""))
            if not ans:
                continue

            cache_key = (qid, ans)
            if cache_key in exact_sc_cache:
                sc, conf, ai_sc = exact_sc_cache[cache_key]
            else:
                vec = emb_of(ans)
                if vec is None:
                    embed_many([ans])
                    vec = emb_of(ans)

                # 1) score vs per-question centroids (FIX)
                sc, conf = None, 0.0
                if qid in q_centroids:
                    sc, conf = score_vec_against_centroids(q_centroids[qid], vec)

                # 2) fallback vs per-attribute centroids
                if sc is None and attr in a_centroids:
                    sc, conf = score_vec_against_centroids(a_centroids[attr], vec)

                # 3) fallback vs global centroids
                if sc is None:
                    sc, conf = score_vec_against_centroids(g_centroids, vec)

                if sc is None:
                    sc = 1
                    conf = 0.0

                # optional low-confidence clamp
                if CONF_CLAMP > 0 and conf < CONF_CLAMP:
                    sc = min(int(sc), 1)

                # off-topic clamp using overlap with hint (conservative)
                if qhint and qa_overlap(ans, qhint) < MIN_QA_OVERLAP:
                    sc = min(int(sc), 1)

                ai_sc = ai_signal_score(ans, qhint)
                exact_sc_cache[cache_key] = (int(sc), float(conf), float(ai_sc))

            if ai_sc >= AI_SUSPECT_THRESHOLD:
                any_ai = True

            # write per-question outputs (Qn1..4)
            qn = None
            if "_Q" in qid:
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None

            if qn in (1, 2, 3, 4):
                row[f"{attr}_Qn{qn}"] = int(sc)
                row[f"{attr}_Rubric_Qn{qn}"] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

        # Fill blanks for consistent columns
        for attr in ORDERED_ATTRS:
            for qn in (1, 2, 3, 4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # Attribute averages + ranks
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                row[f"{attr}_Avg (0–3)"] = ""
                row[f"{attr}_RANK"] = ""
            else:
                avg = float(np.mean(scores))
                band = int(np.clip(int(round(avg)), 0, 3))
                overall_total += band
                row[f"{attr}_Avg (0–3)"] = round(avg, 2)
                row[f"{attr}_RANK"] = BANDS[band]

        row["Overall Total (0–21)"] = overall_total
        row["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        row["AI_suspected"] = bool(any_ai)

        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # Column order (same layout style)
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

    ordered += [c for c in ["Overall Total (0–21)", "Overall Rank", "AI_suspected"] if c in res.columns]
    return res.reindex(columns=[c for c in ordered if c in res.columns])


# =============================================================================
# EXPORTS
# =============================================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return bio.getvalue()


# =============================================================================
# GOOGLE SHEETS PUSH (optional)
# =============================================================================
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
    if not key:
        raise ValueError("GSHEETS_SPREADSHEET_KEY not set in secrets.")
    gc = gs_client()
    sh = gc.open_by_key(key)
    try:
        return sh.worksheet(DEFAULT_WS_NAME)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=DEFAULT_WS_NAME, rows="20000", cols="150")


def upload_df_to_gsheets(df: pd.DataFrame) -> Tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        header = df.columns.astype(str).tolist()
        values = df.astype(object).where(pd.notna(df), "").values.tolist()
        ws.clear()
        ws.spreadsheet.values_batch_update(
            body={
                "valueInputOption": "USER_ENTERED",
                "data": [{"range": f"'{ws.title}'!A1", "values": [header] + values}],
            }
        )
        return True, f"✅ Wrote {len(values)} rows × {len(header)} cols to '{ws.title}'."
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
            <div class="pill">Thought Leadership • Fast Meaning Scoring</div>
            <h1>Thought Leadership</h1>
            <p class="app-header-subtitle">
                Kobo import → per-question centroids (0–3) → rubric bands → exports.
                <br/>No cross-encoder. No bucket-mixing. Meaning-first.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.markdown("### Controls")
    st.sidebar.write(f"Embedder: `{st.secrets.get('EMBEDDER_NAME', 'all-MiniLM-L6-v2')}`")
    st.sidebar.write(f"Softmax temp: `{SOFTMAX_TEMP}`")
    st.sidebar.write(f"Min QA overlap: `{MIN_QA_OVERLAP}`")
    st.sidebar.write(f"Confidence clamp: `{CONF_CLAMP}` (0 = off)")

    # 1) Fetch Kobo FIRST (fast feedback)
    with st.spinner("Fetching Kobo submissions…"):
        df = fetch_kobo_dataframe()

    if df.empty:
        st.warning("No Kobo submissions found (or Kobo credentials missing).")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Fetched dataset")
    st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns):,}")
    st.dataframe(df, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # 2) Load mapping + exemplars
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

    # 3) Build per-question centroids (the fix)
    with st.spinner("Building per-question centroids (cached embedder)…"):
        q_centroids, a_centroids, g_centroids = build_question_centroids(exemplars)

    # Optional debug
    if st.sidebar.checkbox("Show centroid coverage (counts)", value=False):
        rows = []
        for qid, pack in q_centroids.items():
            cnts = pack["counts"].tolist()
            rows.append({"question_id": qid, "n0": cnts[0], "n1": cnts[1], "n2": cnts[2], "n3": cnts[3], "total": sum(cnts)})
        dbg = pd.DataFrame(rows).sort_values(["total", "question_id"], ascending=[True, True])
        st.sidebar.dataframe(dbg, width="stretch")

    # 4) Score
    with st.spinner("Scoring submissions (per question_id)…"):
        scored = score_dataframe(df, mapping, q_centroids, a_centroids, g_centroids)

    st.success("✅ Scoring complete.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Scored table")
    st.dataframe(scored, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⬇️ Export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Excel",
            data=to_excel_bytes(scored),
            file_name="ThoughtLeadership_Scored.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with c2:
        st.download_button(
            "Download CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="ThoughtLeadership_Scored.csv",
            mime="text/csv",
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if AUTO_PUSH:
        with st.spinner("📤 Sending to Google Sheets…"):
            ok, msg = upload_df_to_gsheets(scored)
        (st.success if ok else st.error)(msg)


if __name__ == "__main__":
    main()
