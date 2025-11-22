# networking.py
# ------------------------------------------------------------
# Kobo → Scored Excel / Google Sheets (Networking & Advocacy)
# Exact layout preserved. Robust column resolution A1..H1.
# Batched embeddings + caching. AI-suspect flag at the end.
# ------------------------------------------------------------

import json, re, unicodedata
from pathlib import Path
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
# ==============================
# CSS (Networking & Advocacy theme)
# ==============================
def inject_css():
    st.markdown("""
        <style>
        :root {
            /* Brand colours */
            --primary: #F26A21;            /* CARE orange */
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

        /* Full app background */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #FFF7ED 0, #F9FAFB 40%, #F3F4F6 100%);
            color: var(--text-main);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #1f2937;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        /* Main container width + spacing */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Global text + headings */
        h1, h2, h3 {
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-main);
        }
        h1 {
            font-size: 2.1rem;
            font-weight: 700;
        }
        h2 {
            margin-top: 1.5rem;
            font-size: 1.3rem;
        }
        p, span, label {
            color: var(--text-muted);
        }

        /* App header card with orange + gold + silver accent */
        .app-header-card {
            position: relative;
            background:
                radial-gradient(circle at top left,
                    rgba(242,106,33,0.15),
                    rgba(250,204,21,0.06),
                    #ffffff);
            border-radius: 1.25rem;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.12);
            margin-bottom: 1.4rem;
            overflow: hidden;
        }

        /* thin gold/silver strip at the top */
        .app-header-card::before {
            content: "";
            position: absolute;
            inset: 0;
            height: 3px;
            background: linear-gradient(90deg,
                var(--gold-soft),
                var(--primary),
                var(--silver),
                var(--gold));
            opacity: 0.95;
        }

        /* soft glow in the corner */
        .app-header-card::after {
            content: "";
            position: absolute;
            bottom: -40px;
            right: -40px;
            width: 140px;
            height: 140px;
            background: radial-gradient(circle,
                rgba(250,204,21,0.35),
                transparent 60%);
            opacity: 0.7;
        }

        .app-header-card h1 {
            margin-bottom: 0.2rem;
        }
        .app-header-subtitle {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .pill {
            display: inline-block;
            font-size: 0.75rem;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            background: rgba(242,106,33,0.08);
            border: 1px solid rgba(242,106,33,0.6);
            color: #9A3412;
            margin-bottom: 0.4rem;
        }

        /* Section “cards” */
        .section-card {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border-subtle);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.04);
        }

        /* Dataframe tables */
        .stDataFrame table {
            font-size: 13px;
            border-radius: 0.75rem;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
        }
        .stDataFrame table thead tr th {
            background-color: var(--primary-soft);
            font-weight: 600;
            color: #7c2d12;
        }

        /* Buttons & download buttons (CARE orange) */
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

        /* Success / warning / error blocks */
        .stAlert {
            border-radius: 0.8rem;
        }

        /* Hide default Streamlit menu + footer for cleaner look (optional) */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

# ==============================
# CONSTANTS / PATHS
# ==============================
KOBO_BASE      = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID3 = st.secrets.get("KOBO_ASSET_ID3", "")
KOBO_TOKEN     = st.secrets.get("KOBO_TOKEN", "")
AUTO_PUSH      = bool(st.secrets.get("AUTO_PUSH", False))
AUTO_RUN       = bool(st.secrets.get("AUTO_RUN", True))

DATASETS_DIR   = Path("DATASETS")
MAPPING_PATH   = DATASETS_DIR / "mapping3.csv"
EXEMPLARS_PATH = DATASETS_DIR / "networking.jsonl"

BANDS = {0: "Counterproductive", 1: "Compliant", 2: "Strategic", 3: "Transformative"}
OVERALL_BANDS = [
    ("Localization Champion",        21, 24),
    ("Skilled Networked Advocate",   16, 20),
    ("Developing Influencer",        10, 15),
    ("Needs Support",                 0,  9),
]

ORDERED_ATTRS = [
    "Strategic Positioning & Donor Fluency",    # A1
    "Power-Aware Stakeholder Mapping",          # B1
    "Equitable Allyship & Local Fronting",      # C1
    "Coalition Governance & Convening",         # D1
    "Community-Centered Messaging",             # E1
    "Evidence-Led Learning (Local Knowledge)",  # F1
    "Influence Without Authority",              # G1
    "Risk Management & Adaptive Communication", # H1
]

# Attribute → header section code (used in your sheet headers like Power_Aware/B1_1)
ATTR_TO_HEADER_SECT = {
    "Strategic Positioning & Donor Fluency":           "A1",
    "Power-Aware Stakeholder Mapping":                 "B1",
    "Equitable Allyship & Local Fronting":             "C1",
    "Coalition Governance & Convening":                "D1",
    "Community-Centered Messaging":                    "E1",
    "Evidence-Led Learning (Local Knowledge)":         "F1",
    "Influence Without Authority":                     "G1",
    "Risk Management & Adaptive Communication":        "H1",
}

# QID prefix → canonical section used in Kobo export paths
# (Your mapping3.csv uses prefixes like SPD_Q#, PAS_Q#, etc.)
QID_PREFIX_TO_SECTIONS = {
    "SPD": ["A1"],
    "PAS": ["B1"],
    "EAL": ["C1"],
    "CGC": ["D1"],
    "CCM": ["E1"],
    "ELL": ["F1"],
    "IWA": ["G1"],
    "RMA": ["H1"],
}

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

# passthrough source columns we keep (front pool; later filtered by explicit exclude set)
PASSTHROUGH_HINTS = [
    "staff id","staff_id","staffid","_id","id","_uuid","uuid","instanceid","_submission_time",
    "submissiondate","submission_date","start","_start","end","_end","today","date","deviceid",
    "username","enumerator","submitted_via_web","_xform_id_string","formid","assetid","care_staff","type_participant"
]
# EXCLUDE these specific raw source cols from the visible table
_EXCLUDE_SOURCE_COLS_LOWER = {
    "_id","formhub/uuid","start","end","today","staff_id","meta/instanceid",
    "_xform_id_string","_uuid","meta/rootuuid","_submission_time","_validation_status"
}

# ==============================
# AI DETECTION (aggressive)
# ==============================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

# ==============================
# AI HEURISTICS (refined)
# ==============================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

TRANSITION_OPEN_RX = re.compile(
    r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b",
    re.I
)
LIST_CUES_RX       = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX          = re.compile(r"^[-*•]\s", re.M)

# Hard trigger: any long dash (en/em)
LONG_DASH_HARD_RX  = re.compile(r"[—–]")

# Symbols (now also picks up __, --- etc.)
SYMBOL_RX = re.compile(
    r"[—–\-_]{2,}"         # runs of dashes or underscores: --, —-, ___
    r"|[≥≤≧≦≈±×÷%]"        # math-ish symbols
    r"|[→←⇒↔↑↓]"           # arrows
    r"|[•●◆▶✓✔✗❌§†‡]",    # bullets / ticks / section marks
    re.U
)

# Time / duration cues
TIMEBOX_RX = re.compile(
    r"(?:\bday\s*\d+\b|\bweek\s*\d+\b|\bmonth\s*\d+\b|\bquarter\s*\d+\b"
    r"|\b\d+\s*(?:days?|weeks?|months?|quarters?)\b|\bby\s+day\s*\d+\b)",
    re.I
)

# Explicit AI confessions
AI_RX = re.compile(r"(?:as an ai\b|i am an ai\b)", re.I)

# Planning / outline patterns
DAY_RANGE_RX        = re.compile(r"\bday\s*\d+\s*[-–]\s*\d+\b", re.I)
PIPE_LIST_RX        = re.compile(r"\s\|\s")  # " | " as section separator
PARENS_ACRONYMS_RX  = re.compile(r"\(([A-Z]{2,}(?:s)?(?:\s*,\s*[A-Z]{2,}(?:s)?)+).*?\)")
# accepts 1.), 2) etc.
NUMBERED_BULLETS_RX = re.compile(r"\b\d+\s*[\.\)]\s*")
SLASH_PAIR_RX       = re.compile(r"\b\w+/\w+\b")  # financially/economically

AI_BUZZWORDS = {
    "minimum viable", "feedback loop", "trade-off", "evidence-based",
    "stakeholder alignment", "learners' agency", "learners’ agency",
    "norm shifts", "quick win", "low-lift", "scalable",
    "best practice", "pilot theatre", "timeboxed"
}

# ==============================
# HELPERS
# ==============================
def clean(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()

def qa_overlap(ans: str, qtext: str) -> float:
    at = set(re.findall(r"\w+", (ans or "").lower()))
    qt = set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

def ai_signal_score(text: str, question_hint: str = "") -> float:
    """
    Heuristic AI-likeness score in [0,1].
    You flag as AI when this >= AI_SUSPECT_THRESHOLD (e.g. 0.60).
    """
    t = clean(text)
    if not t:
        return 0.0

    # HARD RULE: any en/em dash anywhere → treat as AI-like
    if LONG_DASH_HARD_RX.search(t):
        return 1.0

    score = 0.0

    # core symbol / structure cues
    if SYMBOL_RX.search(t):               score += 0.35
    if TIMEBOX_RX.search(t):              score += 0.15
    if AI_RX.search(t):                   score += 0.35
    if TRANSITION_OPEN_RX.search(t):      score += 0.12
    if LIST_CUES_RX.search(t):            score += 0.12
    if BULLET_RX.search(t):               score += 0.08

    # planning / outline patterns
    if DAY_RANGE_RX.search(t):            score += 0.15   # "Day 1-10"
    if PIPE_LIST_RX.search(t):            score += 0.10   # " | "
    if PARENS_ACRONYMS_RX.search(t):      score += 0.10   # (FGDs, KII, ...)
    if NUMBERED_BULLETS_RX.search(t):     score += 0.12   # 1.), 2)
    if SLASH_PAIR_RX.search(t):           score += 0.08   # financially/economically

    # combo bonus: if it really looks like a structured plan
    hits = 0
    for rx in (TIMEBOX_RX, DAY_RANGE_RX, PIPE_LIST_RX, NUMBERED_BULLETS_RX):
        if rx.search(t):
            hits += 1
    if hits >= 2:
        score += 0.25
    if hits >= 3:
        score += 0.15

    # buzzwords
    tl = t.lower()
    buzz_hits = sum(1 for b in AI_BUZZWORDS if b in tl)
    if buzz_hits:
        score += min(0.24, 0.08 * buzz_hits)

    # off-topic generic answer → nudge up
    if question_hint:
        overlap = qa_overlap(t, question_hint)
        if overlap < 0.06:
            score += 0.10

    # clamp to [0,1]
    return max(0.0, min(1.0, score))


# ==============================
# KOBO
# ==============================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID3 or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID3 and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions","data"):
        url = kobo_url(KOBO_ASSET_ID3, kind)
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
            if r.status_code == 404:
                continue
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()
    st.error("Could not fetch data. Check KOBO_BASE, KOBO_ASSET_ID3, token permissions.")
    return pd.DataFrame()

# ==============================
# MAPPING + EXEMPLARS
# ==============================
QNUM_RX = re.compile(r"_Q(\d+)$")

def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), \
        "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def build_bases_from_qid(question_id: str) -> list[str]:
    """Return plausible Kobo base prefixes for a given qid across Networking schema."""
    out = []
    if not question_id: return out
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m: return out
    qn = m.group(1)
    pref = qid.split("_Q")[0]
    sects = QID_PREFIX_TO_SECTIONS.get(pref, [])
    roots = ["networking", "Networking"]
    for sect in sects:
        for root in roots:
            out.append(f"{root}/{sect}_Section/{sect}_{qn}")
    return out

def expand_possible_kobo_columns(base: str) -> list[str]:
    if not base: return []
    return [
        base,
        f"{base} :: Answer (text)",
        f"{base} :: English (en)",
        f"{base} - English (en)",
        f"{base}_labels",
        f"{base}_label",
    ]

def _score_kobo_header(col: str, token: str) -> int:
    c = col.lower(); t = token.lower()
    if c == t: return 100
    s = 0
    if c.endswith("/"+t): s = max(s,95)
    if f"/{t}/" in c: s = max(s,92)
    if f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c: s = max(s,90)
    if t in c: s = max(s,80)
    if "english" in c or "label" in c or "(en)" in c: s += 5
    if "answer (text)" in c or "answer_text" in c or "text" in c: s += 5
    if "networking/" in c or "/a" in c or "/b" in c or "/c" in c or "/d" in c or "/e" in c or "/f" in c or "/g" in c or "/h" in c:
        s += 2
    return s

def resolve_kobo_column_for_mapping(
    df_cols: list[str],
    question_id: str,
    prompt_hint: str,
    attribute: str = ""
) -> str | None:
    """
    Robust mapping: tries canonical path from QID, then A1..H1 token, then fuzzy rescue by prompt_hint.
    Works with headers like 'Power_Aware/B1_1', 'Community_Centered_Messaging/E1_3', etc.
    """
    cols_original = list(df_cols)
    cols_lower = {c.lower(): c for c in cols_original}

    # 1) Canonical path candidates from question_id (Networking/AX_Section/AX_n …)
    bases = build_bases_from_qid(question_id)
    for base in bases:
        lb = base.lower()
        if lb in cols_lower:
            return cols_lower[lb]
        for v in expand_possible_kobo_columns(base):
            lv = v.lower()
            if lv in cols_lower:
                return cols_lower[lv]
        for c in cols_original:  # prefix match
            if c.lower().startswith(lb):
                return c

    # 2) Token candidates derived from QID and from attribute header code (A1..H1)
    tokens = []
    qid = (question_id or "").strip().upper()
    m = QNUM_RX.search(qid)
    qn = m.group(1) if m else None
    if qn:
        pref = qid.split("_Q")[0] if "_Q" in qid else qid
        for sect in QID_PREFIX_TO_SECTIONS.get(pref, []):
            tokens.append(f"{sect}_{qn}")
        hdr = ATTR_TO_HEADER_SECT.get(attribute or "", "")
        if hdr:
            tokens.append(f"{hdr}_{qn}")

    # direct containment (fast path for 'Equitable_Allyship/C1_2', etc.)
    for tok in tokens:
        lt = tok.lower()
        for c in cols_original:
            if lt in c.lower():
                return c

    # scored similarity
    best, bs = None, 0
    for tok in tokens:
        for c in cols_original:
            sc = _score_kobo_header(c, tok)
            if sc > bs:
                bs, best = sc, c
    if best and bs >= 80:
        return best

    # 3) Fuzzy rescue via prompt hint
    hint = clean(prompt_hint or "")
    if hint:
        cands = [(c, c.lower()) for c in cols_original]
        hits = process.extract(hint.lower(), [lo for _, lo in cands],
                               scorer=fuzz.partial_token_set_ratio, limit=5)
        for _, lo, score in hits:
            if score >= 88:
                for orig, low in cands:
                    if low == lo:
                        return orig

    return None

# ==============================
# EMBEDDINGS / CENTROIDS
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def _embed_texts_cached(texts_tuple: tuple[str, ...]) -> dict:
    texts = list(texts_tuple)
    embs = get_embedder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return {t: e for t, e in zip(texts, embs)}

_EMB_CACHE: dict[str, np.ndarray] = {}

def embed_many(texts: list[str]) -> None:
    missing = [t for t in texts if t and t not in _EMB_CACHE]
    if not missing: return
    pack = _embed_texts_cached(tuple(missing))
    _EMB_CACHE.update(pack)

def emb_of(text: str):
    t = clean(text)
    return _EMB_CACHE.get(t, None)

def build_centroids(exemplars: list[dict]):
    by_qkey, by_attr, question_texts = {}, {}, []
    all_texts = []
    for e in exemplars:
        qid   = clean(e.get("question_id",""))
        qtext = clean(e.get("question_text",""))
        score = int(e.get("score",0))
        text  = clean(e.get("text",""))
        attr  = clean(e.get("attribute",""))
        if not qid and not qtext:
            continue
        key = qid if qid else qtext
        if key not in by_qkey:
            by_qkey[key] = {"attribute": attr, "question_text": qtext, "scores": [], "texts": []}
            if qtext: question_texts.append(qtext)
        by_qkey[key]["scores"].append(score)
        by_qkey[key]["texts"].append(text)
        by_attr.setdefault(attr, {0:[],1:[],2:[],3:[]})
        by_attr[attr][score].append(text)
        if text: all_texts.append(text)

    # batch embed exemplar texts once
    embed_many(list(set(all_texts)))

    def centroid(texts):
        vecs = [emb_of(t) for t in texts if emb_of(t) is not None]
        if not vecs: return None
        c = np.stack(vecs, axis=0).mean(axis=0)
        n = np.linalg.norm(c) or 1.0
        return c / n

    def centroids_for_q(texts, scores):
        buckets = {0:[],1:[],2:[],3:[]}
        for t,s in zip(texts, scores):
            if t: buckets[int(s)].append(t)
        return {sc: centroid(batch) for sc, batch in buckets.items()}

    q_centroids    = {k: centroids_for_q(v["texts"], v["scores"]) for k, v in by_qkey.items()}
    attr_centroids = {a: {sc: centroid(txts) for sc, txts in bucks.items()} for a, bucks in by_attr.items()}

    global_buckets = {0:[],1:[],2:[],3:[]}
    for e in exemplars:
        sc = int(e.get("score",0)); txt = clean(e.get("text",""))
        if txt: global_buckets[sc].append(txt)
    global_centroids = {sc: centroid(txts) for sc, txts in global_buckets.items()}

    return q_centroids, attr_centroids, global_centroids, by_qkey, question_texts

def resolve_qkey(q_centroids, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in q_centroids:
        return qid
    hint = clean(prompt_hint or "")
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio) if (hint and question_texts) else None
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted:
                return k
    return None

# ==============================
# SCORING
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts):

    df_cols = list(df.columns)

    # quick visibility
    with st.expander("🔎 Networking-like columns found", expanded=False):
        sample_cols = [c for c in df_cols if "/A" in c or "/B" in c or "/C" in c or "/D" in c or "/E" in c or "/F" in c or "/G" in c or "/H" in c or "Networking/" in c]
        st.write(sample_cols[:120])

    # id/date columns
    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id","staff_id","staffid")), None)
    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end","_submission_time","SubmissionDate","submissiondate"] if c in df.columns), None)

    n_rows = len(df)

    # --- Clean & parse Date / Start / End (match leadership/advisory style) ---
    if date_col in df.columns:
        date_clean = (
            df[date_col]
            .astype(str)
            .str.strip()
            .str.lstrip(",")
        )
        dt_series = pd.to_datetime(date_clean, errors="coerce")
    else:
        dt_series = pd.Series([pd.NaT] * n_rows)

    if start_col:
        start_clean = (
            df[start_col]
            .astype(str)
            .str.strip()
            .str.lstrip(",")
        )
        start_dt = pd.to_datetime(start_clean, utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)

    if end_col:
        end_clean = (
            df[end_col]
            .astype(str)
            .str.strip()
            .str.lstrip(",")
        )
        end_dt = pd.to_datetime(end_clean, utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT] * n_rows)

    # --- Duration in minutes (whole number, clipped at 0) ---
    duration_min = (end_dt - start_dt).dt.total_seconds() / 60.0
    duration_min = duration_min.clip(lower=0)

    # mapping rows we actually use
    rows_out = []
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

    # resolve Kobo headers for each mapping row (now attribute-aware, A1..H1)
    resolved_for_qid = {}
    missing_map_rows = []
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        attr  = r.get("attribute","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint, attr)
        if hit:
            resolved_for_qid[qid] = hit
        else:
            missing_map_rows.append((qid, attr, qhint))

    with st.expander("🧭 Mapping → Kobo column resolution (by question_id)", expanded=False):
        if resolved_for_qid:
            show = [(k, v) for k, v in list(resolved_for_qid.items())[:100]]
            st.dataframe(pd.DataFrame(show, columns=["question_id","kobo_column"]))
        if missing_map_rows:
            st.warning(f"{len(missing_map_rows)} mapping rows not found in headers (showing up to 40).")
            st.dataframe(pd.DataFrame(missing_map_rows[:40], columns=["question_id","attribute","prompt_hint"]))

    # batch-embed all distinct free-text answers once (big speed gain)
    distinct_answers = set()
    for _, row in df.iterrows():
        for r in all_mapping:
            col = resolved_for_qid.get(r["question_id"])
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a: distinct_answers.add(a)
    embed_many(list(distinct_answers))

    rows_out = []
    for i, resp in df.iterrows():
        out = {}
        out["Date"] = (
            pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(dt_series.iloc[i]) else str(i)
        )
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        d_val = duration_min.iloc[i]
        out["Duration_min"] = int(round(d_val)) if not pd.isna(d_val) else ""

        per_attr = {}
        ai_scores = []

        # cache row answers to avoid re-cleaning
        row_answers = {}
        for r in all_mapping:
            qid = r["question_id"]
            dfcol = resolved_for_qid.get(qid)
            if dfcol and dfcol in df.columns:
                row_answers[qid] = clean(resp.get(dfcol, ""))

        # prefetch qtexts for hints (only when needed)
        qtext_cache = {}

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            dfcol = resolved_for_qid.get(qid)
            if not dfcol or dfcol not in df.columns:
                continue

            ans = row_answers.get(qid, "")
            if not ans:
                continue

            vec = emb_of(ans)

            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)
            if qkey and qkey not in qtext_cache:
                qtext_cache[qkey] = (by_qkey.get(qkey, {}) or {}).get("question_text","")
            qtext_full = qtext_cache.get(qkey, "") if qkey else qhint

            sc = None
            if vec is not None:
                def best_sim(cent_dict):
                    best_s, best_v = None, -1e9
                    for s, v in (cent_dict or {}).items():
                        if v is None: continue
                        sim = float(np.dot(vec, v))
                        if sim > best_v:
                            best_v, best_s = sim, s
                    return best_s

                if qkey and qkey in q_centroids:
                    sc = best_sim(q_centroids[qkey])
                if sc is None and attr in attr_centroids:
                    sc = best_sim(attr_centroids[attr])
                if sc is None:
                    sc = best_sim(global_centroids)

                # Penalize off-topic vs question/hint
                if sc is not None and qa_overlap(ans, qtext_full or qhint) < MIN_QA_OVERLAP:
                    sc = min(sc, 1)

            # AI suspicion for this answer
            ai_scores.append(ai_signal_score(ans, qtext_full))

            # write per-question score & rubric Q1..Q4
            qn = None
            if "_Q" in (qid or ""):
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None
            if qn not in (1,2,3,4):
                continue

            score_key  = f"{attr}_Qn{qn}"
            rubric_key = f"{attr}_Rubric_Qn{qn}"
            if sc is None:
                out.setdefault(score_key, "")
                out.setdefault(rubric_key, "")
            else:
                sc_int = int(sc)
                out[score_key]  = sc_int
                out[rubric_key] = BANDS[sc_int]
                per_attr.setdefault(attr, []).append(sc_int)

        # ensure fixed shape for per-question blocks
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + ranks
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                out[f"{attr}_Avg (0–3)"] = ""
                out[f"{attr}_RANK"]      = ""
            else:
                avg = float(np.mean(scores))
                band = int(round(avg))
                overall_total += band
                out[f"{attr}_Avg (0–3)"] = round(avg, 2)
                out[f"{attr}_RANK"]      = BANDS[band]

        out["Overall Total (0–24)"] = overall_total
        out["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        out["AI-Suspected"] = bool(any(s >= AI_SUSPECT_THRESHOLD for s in ai_scores))

        rows_out.append(out)

    res_df = pd.DataFrame(rows_out)

    # Final column order: Date, Staff ID, Duration_min → per-question → per-attr → Overall → AI (last)
    def order_cols(cols):
        ordered = ["Date","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI-Suspected"]
        extras = [c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols] + extras

    return res_df.reindex(columns=order_cols(list(res_df.columns)))

# ==============================
# EXPORTS / SHEETS
# ==============================
def _ensure_ai_last(df: pd.DataFrame,
                    export_name: str = "AI-Suspected",
                    legacy_name: str = "AI_suspected") -> pd.DataFrame:
    out = df.copy()
    if export_name not in out.columns and legacy_name in out.columns:
        out = out.rename(columns={legacy_name: export_name})
    if export_name not in out.columns:
        out[export_name] = ""
    cols = [c for c in out.columns if c != export_name] + [export_name]
    return out[cols]

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    df_out = _ensure_ai_last(df)
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False)
    return bio.getvalue()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME3", "Networking")

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
    required = ["type","project_id","private_key_id","private_key","client_email","client_id","token_uri"]
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
    try:
        sh = gc.open_by_key(key)
    except gspread.SpreadsheetNotFound:
        raise ValueError(f"Spreadsheet with key '{key}' not found or not shared with the service account.")
    try:
        return sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        st.warning(f"Worksheet '{ws_name}' not found. Creating it…")
        return sh.add_worksheet(title=ws_name, rows="20000", cols="200")

def _to_a1_col(n: int) -> str:
    s = []
    while n > 0:
        n, r = divmod(n - 1, 26)
        s.append(chr(65 + r))
    return ''.join(reversed(s))

def _chunk(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def _post_write_formatting(ws: gspread.Worksheet, cols: int) -> None:
    try: ws.freeze(rows=1)
    except Exception: pass
    try:
        ws.spreadsheet.batch_update({
            "requests": [{
                "autoResizeDimensions": {
                    "dimensions": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": int(cols)}
                }
            }]
        })
    except Exception: pass

def upload_df_to_gsheets(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        df = _ensure_ai_last(df)  # ensure AI-Suspected is last
        header = df.columns.astype(str).tolist()
        values = df.astype(object).where(pd.notna(df), "").values.tolist()
        all_rows = [header] + values
        ws.clear()
        col_end = _to_a1_col(len(header))
        data_payload, start_row = [], 1
        for rows in _chunk(all_rows, 10000):
            end_row = start_row + len(rows) - 1
            a1_range = f"'{ws.title}'!A{start_row}:{col_end}{end_row}"
            data_payload.append({"range": a1_range, "values": rows})
            start_row = end_row + 1
        ws.spreadsheet.values_batch_update(body={"valueInputOption":"USER_ENTERED","data":data_payload})
        _post_write_formatting(ws, len(header))
        return True, f"✅ Wrote {len(values)} rows to '{ws.title}' via batch update"
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"


# ==============================
# PAGE ENTRYPOINT (main)
# ==============================
def main():
    inject_css()

    st.markdown("""
        <div class="app-header-card">
            <div class="pill">Networking & Advocacy • Auto Scoring</div>
            <h1>Networking & Advocacy</h1>
            <p class="app-header-subtitle">
                Importing Kobo submissions, scoring networking and advocacy attributes against exemplars,
                flagging AI-like responses, and exporting results to Excel or Google Sheets.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # local, stable DF signature to guard session_state writes
    def _df_sig_local(df: pd.DataFrame) -> str:
        import hashlib
        try:
            h = pd.util.hash_pandas_object(df, index=True).values
            return hashlib.sha1(h.tobytes()).hexdigest()
        except Exception:
            return hashlib.sha1(df.to_csv(index=True).encode("utf-8")).hexdigest()

    # one-time init of session keys (prevents first-run churn)
    for k, v in {
        "scored_df_networking": None,
        "scored_sig_networking": None,
        "excel_bytes_networking": b"",
    }.items():
        st.session_state.setdefault(k, v)

    def run_pipeline():
        # 1) mapping + exemplars
        try:
            mapping = load_mapping_from_path(MAPPING_PATH)
        except Exception as e:
            st.error(f"Failed to load mapping from {MAPPING_PATH}: {e}")
            return

        try:
            exemplars = read_jsonl_path(EXEMPLARS_PATH)
            if not exemplars:
                st.error(f"Exemplars file is empty: {EXEMPLARS_PATH}")
                return
        except Exception as e:
            st.error(f"Failed to read exemplars from {EXEMPLARS_PATH}: {e}")
            return

        with st.spinner("Building semantic centroids..."):
            q_centroids, attr_centroids, global_centroids, by_qkey, question_texts = build_centroids(exemplars)

        # 2) fetch Kobo
        with st.spinner("Fetching Kobo submissions..."):
            df = fetch_kobo_dataframe()
        if df.empty:
            st.warning("No Kobo submissions found.")
            return

        # --- Section: raw dataset ---
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📥 Fetched dataset")
        st.caption(f"Rows: {len(df):,}  •  Columns: {len(df.columns):,}")
        st.dataframe(df.head(), use_container_width=True, height=360)
        st.markdown('</div>', unsafe_allow_html=True)

        # 3) score
        with st.spinner("Scoring responses (+ AI detection)..."):
            scored_df = score_dataframe(
                df, mapping,
                q_centroids, attr_centroids, global_centroids,
                by_qkey, question_texts
            )

        st.success("✅ Scoring complete.")

        # --- highlight AI-suspected rows ---
        def _highlight_ai(row):
            if "AI-Suspected" in row and row["AI-Suspected"]:
                return ["background-color: #241E4E"] * len(row)
            return [""] * len(row)

        styled = scored_df.style.apply(_highlight_ai, axis=1)

        # --- Section: scored table ---
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Scored table")
        st.caption(
            "Date → Staff ID → Duration_min, then per-question scores & rubrics, "
            "attribute averages, Overall score, Overall Rank, and AI-Suspected as the final column."
        )
        st.dataframe(styled, use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- only update session state if results actually changed ---
        sig = _df_sig_local(scored_df)
        if st.session_state["scored_sig_networking"] != sig:
            st.session_state["scored_df_networking"]  = scored_df
            st.session_state["scored_sig_networking"] = sig
            st.session_state["excel_bytes_networking"] = to_excel_bytes(scored_df)

        # --- Section: downloads ---
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⬇️ Export")
        st.caption("Download the scored Networking & Advocacy results for further analysis or sharing.")
        st.download_button(
            "Download Excel",
            data=st.session_state["excel_bytes_networking"],
            file_name="Networking_Scoring.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_xlsx_networking",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # 5) auto-push if configured
        if AUTO_PUSH:
            with st.spinner("📤 Sending to Google Sheets..."):
                ok, msg = upload_df_to_gsheets(scored_df)
            (st.success if ok else st.error)(msg)

    # Auto-run or manual button
    if AUTO_RUN and not st.session_state.get("networking_auto_ran_once"):
        st.session_state["networking_auto_ran_once"] = True
        run_pipeline()
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True, key="btn_run_networking"):
            run_pipeline()
        st.markdown('</div>', unsafe_allow_html=True)

    # Manual push panel (only if we didn't auto-push already)
    if (st.session_state["scored_df_networking"] is not None) and (not AUTO_PUSH):
        with st.expander("Google Sheets export", expanded=True):
            st.write("Spreadsheet key:", st.secrets.get("GSHEETS_SPREADSHEET_KEY") or "⚠️ Not set")
            st.write("Worksheet name:", DEFAULT_WS_NAME)
            if st.button("📤 Send scored table to Google Sheets", use_container_width=True, key="btn_push_networking"):
                ok, msg = upload_df_to_gsheets(st.session_state["scored_df_networking"])
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

if __name__ == "__main__":
    main()
