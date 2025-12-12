# file: leadership.py — Thought Leadership Scoring (auto-run, full source columns, Date→Duration→Care_Staff first, AI last)

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

import json, re, unicodedata
from io import BytesIO
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
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
# SECRETS / PATHS
# ==============================
KOBO_BASE       = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID1  = st.secrets.get("KOBO_ASSET_ID1", "")
KOBO_TOKEN      = st.secrets.get("KOBO_TOKEN", "")
AUTO_PUSH       = bool(st.secrets.get("AUTO_PUSH", False))
AUTO_RUN        = True  # no buttons

DATASETS_DIR    = Path("DATASETS")
MAPPING_PATH    = DATASETS_DIR / "mapping1.csv"
EXEMPLARS_PATH  = DATASETS_DIR / "thought_leadership.cleaned.jsonl"

# ==============================
# CONSTANTS
# ==============================
BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 19, 21),
    ("Strategic Advisor",       14, 18),
    ("Emerging Advisor",        8, 13),
    ("Needs Capacity Support",   0,  7),
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
FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

DUP_SIM        = float(st.secrets.get("DUP_SIM", 0.92))  # reuse score if answer is semantically same for a question
# passthrough source columns we keep (front of table)
PASSTHROUGH_HINTS = [
    "staff id","staff_id","staffid","_id","id","_uuid","uuid","instanceid","_submission_time",
    "submissiondate","submission_date","start","_start","end","_end","today","date","deviceid",
    "username","enumerator","submitted_via_web","_xform_id_string","formid","assetid"
]

# EXCLUDE these specific raw source cols from the visible table (your list)
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
    # treat NaN as empty
    try:
        if isinstance(s, float) and s != s:
            return ""
    except Exception:
        pass

    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def qa_overlap(ans, qtext) -> float:
    # handle NaN/None/etc safely
    def _t(x) -> str:
        if x is None:
            return ""
        # NaN check: NaN != NaN is True
        try:
            if isinstance(x, float) and x != x:
                return ""
        except Exception:
            pass
        return str(x)

    ans_s = _t(ans).lower()
    q_s   = _t(qtext).lower()

    at = set(re.findall(r"\w+", ans_s))
    qt = set(re.findall(r"\w+", q_s))
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
    if not KOBO_ASSET_ID1 or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID1 and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions","data"):
        url = kobo_url(KOBO_ASSET_ID1, kind)
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 404: continue
            r.raise_for_status()
            payload = r.json()
            results = payload if isinstance(payload, list) else payload.get("results", [])
            if not results and "results" not in payload: results = payload
            df = pd.DataFrame(results)
            if not df.empty: df.columns = [str(c).strip() for c in df.columns]
            return df
        except requests.HTTPError:
            if r.status_code in (401,403):
                st.error("Kobo auth failed: check KOBO_TOKEN and tenant."); return pd.DataFrame()
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}"); return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}"); return pd.DataFrame()
    st.error("Could not fetch Kobo data. Check KOBO_BASE/ASSET/TOKEN.")
    return pd.DataFrame()

# ==============================
# MAPPING + EXEMPLARS
# ==============================
QID_PREFIX_TO_SECTION = {"LAV":"A1","II":"A2","EP":"A3","CFC":"A4","FTD":"A5","LDA":"A6","RDM":"A7"}
QNUM_RX = re.compile(r"_Q(\d+)$")

def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), "mapping needs: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    norm = lambda s: re.sub(r"\s+"," ", str(s).strip().lower())
    target = {norm(a): a for a in ORDERED_ATTRS}
    def snap_attr(a):
        key = norm(a)
        if key in target: return target[key]
        best = process.extractOne(key, list(target.keys()), scorer=fuzz.token_set_ratio)
        return target[best[0]] if best and best[1] >= 75 else None
    m["attribute"] = m["attribute"].apply(snap_attr)
    m = m[m["attribute"].notna()].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists(): raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s: rows.append(json.loads(s))
    return rows

def build_kobo_base_from_qid(question_id: str) -> list[str] | None:
    if not question_id: return None
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m: return None
    qn = m.group(1)
    prefix = qid.split("_Q")[0]
    sect = QID_PREFIX_TO_SECTION.get(prefix)
    if not sect: return None
    token = f"{sect}_{qn}"
    roots = ["Thought Leadership", "Leadership"]
    return [f"{root}/{sect}_Section/{token}" for root in roots]

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
    if "thought leadership/" in c or "leadership/" in c or "/a" in c: s += 2
    return s

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    bases = build_kobo_base_from_qid(question_id) or []
    variants = []
    for base in bases:
        variants.extend(expand_possible_kobo_columns(base))
    for v in variants:
        if v in df_cols: return v
    for c in df_cols:
        if any(c.startswith(b) for b in bases): return c
    token = None
    if question_id:
        qid = question_id.strip().upper()
        m = QNUM_RX.search(qid)
        if m:
            qn = m.group(1); prefix = qid.split("_Q")[0]
            sect = QID_PREFIX_TO_SECTION.get(prefix)
            if sect: token = f"{sect}_{qn}"
    if token:
        best, bs = None, 0
        for c in df_cols:
            sc = _score_kobo_header(c, token)
            if sc > bs: bs, best = sc, c
        if best and bs >= 82: return best
    hint = clean(prompt_hint or "")
    if hint:
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=5)
        for col, score, _ in hits:
            if score >= 80: return col
    return None

# ==============================
# EMBEDDINGS (batched, cached)
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def _embed_texts_cached(texts_tuple: tuple[str, ...]) -> dict:
    texts = list(texts_tuple)
    embs = get_embedder().encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
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
        if not qid and not qtext: continue
        key = qid if qid else qtext
        if key not in by_qkey:
            by_qkey[key] = {"attribute": attr, "question_text": qtext, "scores": [], "texts": []}
            if qtext: question_texts.append(qtext)
        by_qkey[key]["scores"].append(score)
        by_qkey[key]["texts"].append(text)
        by_attr.setdefault(attr, {0:[],1:[],2:[],3:[]})
        by_attr[attr][score].append(text)
        if text: all_texts.append(text)

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
    if qid and qid in q_centroids: return qid
    hint = clean(prompt_hint or "")
    if not (hint and question_texts): return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted: return k
    return None

# ==============================
# SCORING (per-question + averages; with passthrough)
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts):

    df_cols = list(df.columns)

    # identify passthrough columns (keep original order)
    def want_col(c):
        lc = c.strip().lower()
        return any(h in lc for h in PASSTHROUGH_HINTS)
    passthrough_cols = [c for c in df_cols if want_col(c)]
    seen = set(); passthrough_cols = [x for x in passthrough_cols if not (x in seen or seen.add(x))]

    # prefer explicit care_staff, fall back to staff id for value
    staff_id_col   = next((c for c in df.columns if c.strip().lower() in ("staff id","staff_id","staffid")), None)
    care_staff_col = next((c for c in df.columns if c.strip().lower() in ("care_staff","care staff","care-staff")), None)

    date_cols_pref = [
        "_submission_time","SubmissionDate","submissiondate",
        "end","End","start","Start","today","date","Date"
    ]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    # Prefer Kobo start/end fields; keep your old fallbacks
    start_col = next((c for c in ["start"] if c in df.columns), None)
    end_col   = next((c for c in ["end"] if c in df.columns), None)

    n_rows = len(df)

    # Clean Date column (in case it also has leading commas)
    if date_col in df.columns:
        date_clean = (
            df[date_col]
            .astype(str)
            .str.strip()
            .str.lstrip(",")     # handle values like ",2025-11-06T08:00:58.431+03:00"
        )
        dt_series = pd.to_datetime(date_clean, errors="coerce")
    else:
        dt_series = pd.Series([pd.NaT] * n_rows)

    # Clean and parse start time
    if start_col:
        start_clean = (
            df[start_col]
            .astype(str)
            .str.strip()
            .str.lstrip(",")     # remove leading comma if present
        )
        start_dt = pd.to_datetime(start_clean, utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)

    # Clean and parse end time
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

    # Exact duration in minutes (including seconds)
        # Duration in minutes (float), then clipped and rounded later per-row
    duration_min = (end_dt - start_dt).dt.total_seconds() / 60.0
    # Avoid negative values if timestamps are odd
    duration_min = duration_min.clip(lower=0)



    # mapping resolution
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]
    resolved_for_qid = {}
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit: resolved_for_qid[qid] = hit

    # batch-embed distinct answers ONCE, but grouped per question (qid)
    distinct_by_qid: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        for r in all_mapping:
            qid = r["question_id"]
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a:
                    distinct_by_qid.setdefault(qid, set()).add(a)

    all_distinct = set()
    for s in distinct_by_qid.values():
        all_distinct |= s
    embed_many(list(all_distinct))

    # caches to enforce within-question consistency (reuse score for paraphrases)
    exact_sc_cache: dict[tuple[str, str], int] = {}   # (qid, answer_text) -> score
    dup_bank: dict[str, list[tuple[np.ndarray, int]]] = {}  # qid -> [(vec, score)]

    out_rows = []
    for i, resp in df.iterrows():
        row = {}

        # Date, Duration, Care_Staff
        row["Date"] = (pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
                       if pd.notna(dt_series.iloc[i]) else str(i))
        val = duration_min.iloc[i]
        row["Duration"] = int(round(val)) if not pd.isna(val) else ""

        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(resp.get(who_col)) if who_col else ""

        # passthrough original source columns (minus excluded list, and not duplicating the first three)
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:  # explicitly drop these
                continue
            if c in ("Date","Duration","Care_Staff"):
                continue
            row[c] = resp.get(c, "")

        per_attr = {}
        any_ai = False
        qtext_cache = {}

        # row answers cache
        row_answers = {}
        for r in all_mapping:
            qid = r["question_id"]; col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                row_answers[qid] = clean(resp.get(col, ""))

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            col = resolved_for_qid.get(qid)
            if not col: continue
            ans = row_answers.get(qid, "")
            if not ans: continue
            vec = emb_of(ans)

            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)
            if qkey and qkey not in qtext_cache:
                qtext_cache[qkey] = (by_qkey.get(qkey, {}) or {}).get("question_text","")
            qhint_full = qtext_cache.get(qkey, "") if qkey else qhint

            was_cached = (qid, ans) in exact_sc_cache
            sc = exact_sc_cache.get((qid, ans))
            reused = False

            # 1) Exact-match cache (fast path), then semantic duplicate reuse within SAME question (qid)
            if sc is None and vec is not None:
                best_dup_sc, best_dup_sim = None, -1.0
                for v2, sc2 in dup_bank.get(qid, []):
                    sim = float(np.dot(vec, v2))  # cosine similarity (embeddings are normalized)
                    if sim > best_dup_sim:
                        best_dup_sim, best_dup_sc = sim, sc2
                if best_dup_sc is not None and best_dup_sim >= DUP_SIM:
                    sc = int(best_dup_sc)
                    reused = True

            # 2) If not a duplicate, score against exemplars
            if sc is None and vec is not None:
                def best_sim(cent_dict):
                    best_s, best_v = None, -1e9
                    for s, c in cent_dict.items():
                        if c is None: continue
                        v = float(np.dot(vec, c))
                        if v > best_v: best_v, best_s = v, s
                    return best_s
                if qkey and qkey in q_centroids:
                    sc = best_sim(q_centroids[qkey])
                if sc is None and attr in attr_centroids:
                    sc = best_sim(attr_centroids[attr])
                if sc is None:
                    sc = best_sim(global_centroids)
                if sc is not None and qa_overlap(ans, qhint_full or qhint) < MIN_QA_OVERLAP:
                    sc = min(sc, 1)

            # cache result for consistency within the same question
            if sc is not None and not was_cached:
                exact_sc_cache[(qid, ans)] = int(sc)
                if vec is not None and not reused:
                    bank = dup_bank.setdefault(qid, [])
                    if len(bank) < 300:
                        bank.append((vec, int(sc)))


            # per-answer AI suspicion
            ai_score = ai_signal_score(ans, qhint_full)
            if ai_score >= AI_SUSPECT_THRESHOLD:
                any_ai = True

            # write per-question score & rubric ONLY for Q1..Q4
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn in (1,2,3,4) and sc is not None:
                sk = f"{attr}_Qn{qn}"
                rk = f"{attr}_Rubric_Qn{qn}"
                row[sk] = int(sc)
                row[rk] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

        # fill missing Qn/rubric fields to keep stable shape
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + ranks
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                row[f"{attr}_Avg (0–3)"] = ""
                row[f"{attr}_RANK"]      = ""
            else:
                avg = float(np.mean(scores)); band = int(round(avg))
                overall_total += band
                row[f"{attr}_Avg (0–3)"] = round(avg, 2)
                row[f"{attr}_RANK"]      = BANDS[band]

        row["Overall Total (0–21)"] = overall_total
        row["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        row["AI_suspected"] = bool(any_ai)
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # ---- final column order ----
    # 1) Date, Duration, Care_Staff
    ordered = [c for c in ["Date","Duration","Care_Staff"] if c in res.columns]

    # 2) All original source columns (original order), minus excluded ones and minus our first three
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date","Duration","Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    # 3) Per-question blocks
    mid_q = []
    for attr in ORDERED_ATTRS:
        for qn in (1,2,3,4):
            mid_q += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
    ordered += [c for c in mid_q if c in res.columns]

    # 4) Attribute avgs + ranks
    mid_a = []
    for attr in ORDERED_ATTRS:
        mid_a += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += [c for c in mid_a if c in res.columns]

    # 5) Overall then AI (AI must be right after Overall Rank AND be the last column)
    ordered += [c for c in ["Overall Total (0–21)","Overall Rank"] if c in res.columns]
    if "AI_suspected" in res.columns:
        ordered += ["AI_suspected"]

    # enforce ordering (extras, if any, are hidden to keep AI last)
    res = res.reindex(columns=ordered)

    return res

# ==============================
# EXPORTS / SHEETS
# ==============================
def _ensure_ai_last(df: pd.DataFrame,
                    export_name: str = "AI_Suspected",
                    source_name: str = "AI_suspected") -> pd.DataFrame:
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

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME1", "Thought Leadership")

def _normalize_sa_dict(raw: dict) -> dict:
    if not raw: raise ValueError("gcp_service_account missing in secrets.")
    sa = dict(raw)
    if "token_ur" in sa and "token_uri" not in sa: sa["token_uri"] = sa.pop("token_ur")
    if sa.get("private_key") and "\\n" in sa["private_key"]:
        sa["private_key"] = sa["private_key"].replace("\\n","\n")
    sa.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    sa.setdefault("auth_uri", "https://accounts.google.com/o/oauth2/auth")
    sa.setdefault("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs")
    required = ["type","project_id","private_key_id","private_key","client_email","client_id","token_uri"]
    missing = [k for k in required if not sa.get(k)]
    if missing: raise ValueError(f"gcp_service_account missing fields: {', '.join(missing)}")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets.get("gcp_service_account"))
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

def _open_ws_by_key() -> gspread.Worksheet:
    key = st.secrets.get("GSHEETS_SPREADSHEET_KEY")
    ws_name = DEFAULT_WS_NAME
    if not key: raise ValueError("GSHEETS_SPREADSHEET_KEY not set in secrets.")
    gc = gs_client()
    sh = gc.open_by_key(key)
    try:
        return sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=ws_name, rows="20000", cols="150")

def _post_write_formatting(ws: gspread.Worksheet, cols: int) -> None:
    try: ws.freeze(rows=1)
    except Exception: pass
    try:
        ws.spreadsheet.batch_update({
            "requests":[{"autoResizeDimensions":{
                "dimensions":{"sheetId": ws.id, "dimension":"COLUMNS","startIndex":0,"endIndex":cols}
            }}]
        })
    except Exception: pass

def upload_df_to_gsheets(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        df_out = _ensure_ai_last(df, export_name="AI_Suspected", source_name="AI_suspected")
        header = df_out.columns.astype(str).tolist()
        values = df_out.astype(object).where(pd.notna(df_out), "").values.tolist()
        ws.clear()
        ws.spreadsheet.values_batch_update(
            body={"valueInputOption":"USER_ENTERED","data":[{"range": f"'{ws.title}'!A1","values":[header]+values}]}
        )
        _post_write_formatting(ws, len(header))
        return True, f"✅ Wrote {len(values)} rows × {len(header)} cols to '{ws.title}' (last='AI_Suspected')."
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"

# ==============================
# MAIN (auto-run, full tables)
# ==============================
def main():
    inject_css()

    # --- Header card ---
    st.markdown("""
        <div class="app-header-card">
            <div class="pill">Thought Leadership • Auto Scoring</div>
            <h1>Thought Leadership</h1>
            <p class="app-header-subtitle">
                Importing Kobo submissions, scoring CARE thought leadership attributes, flagging AI-like responses,
                and exporting results to Google Sheets.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # mapping + exemplars
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

    with st.spinner("Building semantic centroids..."):
        q_c, a_c, g_c, by_q, qtexts = build_centroids(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found.")
        return

    # --- Section: Raw fetched dataset ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Fetched dataset")
    st.caption(f"Rows: {len(df):,}  •  Columns: {len(df.columns):,}")
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Section: Scored table ---
    with st.spinner("Scoring (+ AI detection)..."):
        scored = score_dataframe(df, mapping, q_c, a_c, g_c, by_q, qtexts)

    st.success("✅ Scoring complete.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Scored table")
    st.caption(
        "Date → Duration → Care_Staff, then source columns (excluded set removed), "
        "per-question scores & rubrics, attribute averages, Overall score, Overall Rank, "
        "and AI_suspected as the final column."
    )

    # highlight AI-suspected rows in soft gold
    def _highlight_ai(row):
        if "AI_suspected" in row and row["AI_suspected"]:
            return ["background-color: #241E4E"] * len(row)
        return [""] * len(row)

    styled = scored.style.apply(_highlight_ai, axis=1)
    st.dataframe(styled, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Section: Downloads ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⬇️ Export")
    st.caption("Download the scored results for further analysis or sharing.")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Excel",
            data=to_excel_bytes(scored),
            file_name="Leadership_Scoring.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=_ensure_ai_last(scored).to_csv(index=False).encode("utf-8"),
            file_name="Leadership_Scoring.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Optional: auto push to Google Sheets
    if AUTO_PUSH:
        with st.spinner("📤 Sending to Google Sheets..."):
            ok, msg = upload_df_to_gsheets(scored)
        (st.success if ok else st.error)(msg)

if __name__ == "__main__":
    main()
