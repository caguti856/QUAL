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

# ==============================
# UI / STYLING
# ==============================
def inject_css():
    st.markdown("""
    <style>
        :root {
            --care-orange: #F26A21;
            --care-orange-soft: rgba(242, 106, 33, 0.10);
            --card-bg: rgba(255, 255, 255, 0.06);
            --text-main: #f5f5f5;
            --text-muted: rgba(245,245,245,0.75);
            --border-soft: rgba(255,255,255,0.15);
        }

        .block-container {
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
            font-size: 1.35rem;
            font-weight: 650;
        }

        /* CARE accent bar */
        .care-accent {
            border-left: 6px solid var(--care-orange);
            padding-left: 14px;
            margin-bottom: 18px;
        }
        .app-header-subtitle {
            margin-top: -8px;
            color: var(--text-muted);
            font-size: 0.98rem;
        }

        /* Cards */
        .section-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 18px 18px 12px 18px;
            margin: 14px 0 18px 0;
            box-shadow: 0 10px 18px rgba(0,0,0,0.25);
        }

        /* Download buttons */
        .stDownloadButton button {
            border-radius: 12px;
            border: 1px solid var(--border-soft);
        }

        /* Dataframe */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-soft);
        }

        /* Metric-like chips */
        .chip {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid var(--border-soft);
            background: rgba(255,255,255,0.05);
            color: var(--text-main);
            font-size: 0.86rem;
            margin-right: 8px;
            margin-bottom: 6px;
        }
        .chip strong { color: var(--care-orange); }

        /* Small helper text */
        .helper {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.35rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# CONFIG (secrets with defaults)
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
    ("Strong Thought Leader",    15, 18),
    ("Developing Thought Leader", 9, 14),
    ("Needs Growth",              0, 8),
]

ORDERED_ATTRS = [
    "Locally Anchored Visioning",
    "Rooted Authority & Voice",
    "Partnering & Power-Sharing",
    "Learning Agility",
    "Systems Thinking",
    "Evidence-Informed Influence",
    "Ethical, Inclusive Digital Practice",
]

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

DUP_SIM        = float(st.secrets.get("DUP_SIM", 0.95))  # reuse score if answer is semantically same for a question
KNN_METHOD     = st.secrets.get("KNN_METHOD", "top1")  # 'top1' or 'softmax'
KNN_K          = int(st.secrets.get("KNN_K", 7))
KNN_TEMP       = float(st.secrets.get("KNN_TEMP", 0.08))
KNN_CONF_MIN   = float(st.secrets.get("KNN_CONF_MIN", 0.45))  # low-confidence clamp threshold

PASSTHROUGH_HINTS = [
    "name", "email", "staff", "unit", "department", "programme", "program", "country",
    "role", "title", "position", "organization", "organisation", "location",
    "submission", "submitted", "date", "timestamp", "duration"
]

# Exclude “admin/system” columns from the output even though we keep “full source columns”
_EXCLUDE_SOURCE_COLS_LOWER = set([
    "_id", "_uuid", "_submission_time", "_submitted_by", "_version",
    "_tags", "_notes", "_status", "_geolocation", "_xform_id_string",
    "_attachments", "_validation_status"
])

# ==============================
# TEXT HELPERS
# ==============================
def clean(s):
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def qa_overlap(answer: str, question_hint: str) -> float:
    a = clean(answer).lower()
    q = clean(question_hint).lower()
    if not a or not q: return 1.0
    a_tokens = set(re.findall(r"[a-z0-9']+", a))
    q_tokens = set(re.findall(r"[a-z0-9']+", q))
    if not q_tokens: return 1.0
    return len(a_tokens & q_tokens) / max(1, len(q_tokens))

# ==============================
# AI-LIKE SIGNAL (lightweight)
# ==============================
def _t(s): return clean(s).lower()

PARENS_ACRONYMS_RX = re.compile(r"\(([A-Z]{2,}|[A-Z]{2,}s)\)")
LEAD_IN_RX = re.compile(r"^(?:sure|certainly|absolutely|here(?:'| i)s)\b", re.I)
FIVE_PAR_RX = re.compile(r"\b(?:1\)|2\)|3\)|4\)|5\))")
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
    r"|(?:\bETA\b|\bKPI\b|\bOKR\b)"  # buzz acronyms
)

def ai_signal_score(answer: str, qhint: str) -> float:
    """
    Heuristic AI-likeness score in [0..1].
    Not a detector; used only as a “suspect” flag.
    """
    t = clean(answer)
    if not t: return 0.0

    score = 0.0
    L = len(t)

    # Overly “template” openings
    if LEAD_IN_RX.search(t):            score += 0.12
    if TRANSITION_OPEN_RX.search(t):    score += 0.10

    # Excessively structured list cues
    if FIVE_PAR_RX.search(t):           score += 0.12
    if len(LIST_CUES_RX.findall(t))>=3: score += 0.10
    if len(BULLET_RX.findall(t))>=4:    score += 0.10

    # Long dash use (common in AI prose)
    if LONG_DASH_HARD_RX.search(t):     score += 0.15

    # Many symbols / “polished” math-ish characters
    if SYMBOL_RX.search(t):             score += 0.08

    # Acronyms in parentheses
    if PARENS_ACRONYMS_RX.search(t):    score += 0.10   # (FGDs, KII, ...)

    # Very long and generic
    if L > 900:                         score += 0.08
    if L > 1400:                        score += 0.10

    # Weak linkage to question hint can add a little
    ov = qa_overlap(t, qhint)
    if ov < 0.04:                       score += 0.08

    return max(0.0, min(1.0, score))

AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

# ==============================
# KOBO PULL
# ==============================
def kobo_url(asset_id: str) -> str:
    return f"{KOBO_BASE}/api/v2/assets/{asset_id}/data/?format=json"

def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID1 or not KOBO_TOKEN:
        st.error("Missing KOBO_ASSET_ID1 or KOBO_TOKEN in secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    url = kobo_url(KOBO_ASSET_ID1)
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    payload = r.json()
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame()
    return pd.json_normalize(results)

# ==============================
# MAPPING + EXEMPLARS
# ==============================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path)
    need = {"question_id","attribute","match_score"}
    if not need.issubset(set(m.columns)):
        raise ValueError(f"mapping file must have columns: {sorted(need)}")
    m["question_id"] = m["question_id"].astype(str).str.strip()
    m["attribute"]   = m["attribute"].astype(str).str.strip()
    m["match_score"] = pd.to_numeric(m["match_score"], errors="coerce")
    m = m.dropna(subset=["match_score"]).copy()
    # snap attributes to known set
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
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

# ==============================
# KOBO COLUMN RESOLUTION (mapping → actual column)
# ==============================
def build_kobo_base_from_qid(qid: str) -> str:
    """
    Kobo exports sometimes flatten fieldnames; this tries to match likely prefixes.
    """
    qid = clean(qid)
    # for "LAV_Q1" try "LAV_Q1" and variations
    return qid

def expand_possible_kobo_columns(qid: str) -> list[str]:
    base = build_kobo_base_from_qid(qid)
    opts = [base]
    # common Kobo patterns: group/name, underscore variations
    opts += [base.lower(), base.upper()]
    opts += [base.replace("_", " "), base.replace("_", "-"), base.replace("-", "_")]
    # strip spaces
    opts += [o.replace(" ", "") for o in opts]
    # de-dup preserve order
    seen=set(); out=[]
    for o in opts:
        if o and o not in seen:
            seen.add(o); out.append(o)
    return out

def _score_kobo_header(candidate: str, target: str) -> int:
    return fuzz.token_set_ratio(clean(candidate), clean(target))

def resolve_kobo_column_for_mapping(df_cols: list[str], qid: str, prompt_hint: str = "") -> str | None:
    """
    Returns the best matching column in df for a given question_id/prompt_hint.
    """
    if not df_cols: return None
    # exact-ish tries
    for opt in expand_possible_kobo_columns(qid):
        for c in df_cols:
            if clean(c).lower() == clean(opt).lower():
                return c

    # fuzzy match with hint if available
    hint = clean(prompt_hint)
    if hint:
        best = process.extractOne(hint, df_cols, scorer=fuzz.token_set_ratio)
        if best and best[1] >= FUZZY_THRESHOLD:
            return best[0]

    # else fuzzy match with qid itself
    best = process.extractOne(qid, df_cols, scorer=fuzz.token_set_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        return best[0]

    return None

# ==============================
# EMBEDDINGS
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

def _pack_from_texts_scores(texts: list[str], scores: list[int], k_key: str, question_text: str = "", attribute: str = ""):
    """Create a scoring pack: embeddings matrix + aligned scores/texts."""
    # Clean + de-dup while preserving order
    seen = set()
    cleaned_texts, cleaned_scores = [], []
    for t, s in zip(texts, scores):
        tt = clean(t)
        if not tt:
            continue
        key = (k_key, tt, int(s))
        if key in seen:
            continue
        seen.add(key)
        cleaned_texts.append(tt)
        cleaned_scores.append(int(s))

    embed_many(list(set(cleaned_texts)))

    vecs, keep_texts, keep_scores = [], [], []
    for t, s in zip(cleaned_texts, cleaned_scores):
        v = emb_of(t)
        if v is None:
            continue
        vecs.append(v)
        keep_texts.append(t)
        keep_scores.append(int(s))

    if not vecs:
        # all-MiniLM-L6-v2 is 384-dim; keep a consistent empty shape
        mat = np.zeros((0, 384), dtype=np.float32)
    else:
        mat = np.vstack(vecs)

    return {
        "vecs": mat,
        "scores": np.array(keep_scores, dtype=int),
        "texts": keep_texts,
        "question_text": clean(question_text),
        "attribute": clean(attribute),
    }

def build_exemplar_index(exemplars: list[dict]):
    """Build per-question exemplar embedding index + fallbacks."""
    by_qkey = {}
    by_attr = {}
    question_texts = []

    # Collect raw texts per key
    for e in exemplars:
        qid   = clean(e.get("question_id", ""))
        qtext = clean(e.get("question_text", ""))
        text  = clean(e.get("text", ""))
        attr  = clean(e.get("attribute", ""))
        try:
            score = int(e.get("score", 0))
        except Exception:
            score = 0

        if not (qid or qtext) or not text:
            continue

        key = qid if qid else qtext

        if key not in by_qkey:
            by_qkey[key] = {"attribute": attr, "question_text": qtext, "scores": [], "texts": []}
            if qtext:
                question_texts.append(qtext)

        by_qkey[key]["scores"].append(score)
        by_qkey[key]["texts"].append(text)

        if attr:
            by_attr.setdefault(attr, {"scores": [], "texts": []})
            by_attr[attr]["scores"].append(score)
            by_attr[attr]["texts"].append(text)

    # Build per-question packs
    ex_index = {}
    for key, pack in by_qkey.items():
        ex_index[key] = _pack_from_texts_scores(
            pack["texts"], pack["scores"],
            k_key=key,
            question_text=pack.get("question_text", ""),
            attribute=pack.get("attribute", ""),
        )

    # Attribute-level packs
    attr_index = {}
    for attr, pack in by_attr.items():
        attr_index[attr] = _pack_from_texts_scores(
            pack["texts"], pack["scores"],
            k_key=f"ATTR::{attr}",
            question_text="",
            attribute=attr,
        )

    # Global pack
    g_texts, g_scores = [], []
    for e in exemplars:
        txt = clean(e.get("text", ""))
        if not txt:
            continue
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0
        g_texts.append(txt)
        g_scores.append(sc)

    global_pack = _pack_from_texts_scores(g_texts, g_scores, k_key="GLOBAL")

    return ex_index, attr_index, global_pack, by_qkey, question_texts

def _topk_matches(pack: dict, ans_vec: np.ndarray, k: int = 7):
    if pack is None or pack.get("vecs") is None or pack["vecs"].size == 0:
        return []

    sims = pack["vecs"] @ ans_vec  # cosine sim (embeddings normalized)
    k = max(1, min(int(k), len(sims)))

    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(float(sims[i]), int(pack["scores"][i]), pack["texts"][i]) for i in top_idx]

def score_vec_against_pack(pack: dict, ans_vec: np.ndarray, k: int = 7, temp: float = 0.08, method: str = "top1"):
    """Score an answer vector against an exemplar pack."""
    matches = _topk_matches(pack, ans_vec, k=k)
    if not matches:
        return None, 0.0, []

    method = (method or "top1").lower().strip()

    if method == "softmax":
        sims_arr = np.array([m[0] for m in matches], dtype=float)
        scores_arr = np.array([m[1] for m in matches], dtype=int)

        w = np.exp((sims_arr - sims_arr.max()) / float(temp))
        w = w / (w.sum() + 1e-9)

        class_w = np.zeros(4, dtype=float)
        for sc, wi in zip(scores_arr, w):
            if 0 <= int(sc) <= 3:
                class_w[int(sc)] += float(wi)

        pred = int(class_w.argmax())
        conf = float(class_w.max())
        return pred, conf, matches

    # Default: score = score of the single closest exemplar
    best_sim, best_sc, _ = matches[0]
    return int(best_sc), float(best_sim), matches

def score_answer_knn(qkey: str, ans_vec: np.ndarray,
                     ex_index: dict, k: int = 7, temp: float = 0.08, method: str = "top1"):
    pack = ex_index.get(qkey)
    return score_vec_against_pack(pack, ans_vec, k=k, temp=temp, method=method)

# ==============================
# (Legacy) centroid builder kept for reference
# ==============================
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

def resolve_qkey(available_keys, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in available_keys: return qid
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
                    ex_index, attr_index, global_pack,
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

    # resolve the "Date" and "Duration" columns if present (Kobo patterns vary)
    date_col = next((c for c in df_cols if c.strip().lower() in ("date", "submission_date", "_submission_time")), None)
    dur_col  = next((c for c in df_cols if "duration" in c.strip().lower()), None)

    # resolve mapping rows to Kobo columns
    resolved_for_qid = {}
    all_mapping = mapping.to_dict("records")
    for r in all_mapping:
        qid = r["question_id"]
        qhint = r.get("column","") or r.get("prompt_hint","") or ""
        col = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if col:
            resolved_for_qid[qid] = col

    # exact match cache: (qid, answer_text) -> score
    exact_sc_cache = {}
    # semantic duplicate bank: qid -> list[(vec, score)]
    dup_bank = {}

    out_rows = []

    # cache for qtext full
    qtext_cache = {}

    # Pre-collect unique answers and embed them in one go
    unique_answers = set()
    for _, resp in df.iterrows():
        for qid, col in resolved_for_qid.items():
            if col and col in df.columns:
                a = clean(resp.get(col, ""))
                if a:
                    unique_answers.add(a)
    embed_many(list(unique_answers))

    for _, resp in df.iterrows():
        row = {}
        # Date / Duration
        if date_col and date_col in df.columns:
            row["Date"] = resp.get(date_col, "")
        if dur_col and dur_col in df.columns:
            row["Duration"] = resp.get(dur_col, "")

        # Care_Staff column: prefer care_staff, else staff id, else blank
        if care_staff_col and care_staff_col in df.columns:
            row["Care_Staff"] = resp.get(care_staff_col, "")
        elif staff_id_col and staff_id_col in df.columns:
            row["Care_Staff"] = resp.get(staff_id_col, "")
        else:
            row["Care_Staff"] = ""

        # Add full passthrough columns
        for c in passthrough_cols:
            if c in (date_col, dur_col, care_staff_col):  # already represented in our first three
                continue
            if c in df.columns:
                row[c] = resp.get(c, "")

        # per-attribute per-question scores
        per_attr = {}
        any_ai = False

        # build row answers dict for this response (qid->answer)
        row_answers = {}
        for qid, col in resolved_for_qid.items():
            if col and col in df.columns:
                row_answers[qid] = clean(resp.get(col, ""))

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            col = resolved_for_qid.get(qid)
            if not col: continue
            ans = row_answers.get(qid, "")
            if not ans: continue
            vec = emb_of(ans)

            qkey = resolve_qkey(ex_index, by_qkey, question_texts, qid, qhint)
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

            # 2) If not a duplicate, score against exemplars (nearest exemplar per question)
            if sc is None and vec is not None:
                # Nearest-exemplar scoring (per question first; then attribute; then global)
                # Method: top-1 exemplar by default (KNN_METHOD='top1'), or softmax vote over top-k.
                sc_knn, conf, matches = None, 0.0, []
                if qkey and qkey in ex_index:
                    sc_knn, conf, matches = score_answer_knn(qkey, vec, ex_index, k=KNN_K, temp=KNN_TEMP, method=KNN_METHOD)
                if sc_knn is None and attr in attr_index:
                    sc_knn, conf, matches = score_vec_against_pack(attr_index[attr], vec, k=KNN_K, temp=KNN_TEMP, method=KNN_METHOD)
                if sc_knn is None:
                    sc_knn, conf, matches = score_vec_against_pack(global_pack, vec, k=KNN_K, temp=KNN_TEMP, method=KNN_METHOD)

                sc = sc_knn

                # Low-confidence clamp (avoid giving high scores when nearest evidence is weak)
                if sc is not None and conf < KNN_CONF_MIN:
                    sc = min(int(sc), 1)

                # Off-topic clamp (question/answer overlap too low)
                if sc is not None and qa_overlap(ans, qhint_full or qhint) < MIN_QA_OVERLAP:
                    sc = min(int(sc), 1)

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
                avg = float(np.mean(scores))
                band = int(np.floor(avg + 1e-9))
                band = max(0, min(3, band))
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
    source_cols = [c for c in source_cols if c not in ordered]
    # If care_staff_col existed, we already used it for Care_Staff; we still keep original if user wants.
    ordered += [c for c in source_cols if c in res.columns and c not in ordered]

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

    # 6) any leftovers
    ordered += [c for c in res.columns if c not in ordered]

    res = res[ordered]
    res = res.rename(columns={"AI_suspected":"AI_Suspected"})
    return res

def _ensure_ai_last(df: pd.DataFrame, export_name="AI_Suspected", source_name="AI_suspected") -> pd.DataFrame:
    cols = list(df.columns)
    if export_name in cols:
        cols = [c for c in cols if c != export_name] + [export_name]
        return df[cols]
    if source_name in cols:
        cols = [c for c in cols if c != source_name] + [source_name]
        return df[cols]
    return df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Scored")
    return output.getvalue()

# ==============================
# GOOGLE SHEETS UPLOAD
# ==============================
def _normalize_sa_dict(sa: dict) -> dict:
    # accept secrets-like dict
    if not sa: return {}
    # streamlit secrets may include private key with \n
    if "private_key" in sa:
        sa = dict(sa)
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets.get("gcp_service_account", {}))
    if not sa:
        return None
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    return gspread.authorize(creds)

def _open_ws_by_key():
    gc = gs_client()
    if gc is None:
        raise RuntimeError("Missing gcp_service_account in secrets.")
    sheet_key = st.secrets.get("GSHEET_KEY", "")
    tab_name  = st.secrets.get("GSHEET_TAB", "Sheet1")
    if not sheet_key:
        raise RuntimeError("Missing GSHEET_KEY in secrets.")
    sh = gc.open_by_key(sheet_key)
    ws = sh.worksheet(tab_name)
    return ws

def _post_write_formatting(ws, n_cols: int):
    # freeze header row
    ws.freeze(rows=1)
    # set filter
    ws.set_basic_filter()
    # set column widths a bit
    widths = [{"range": {"sheetId": ws._properties["sheetId"], "startColumnIndex": i, "endColumnIndex": i+1},
               "pixelSize": 180} for i in range(min(n_cols, 40))]
    body = {"requests": [{"updateDimensionProperties":{
        "range": w["range"],
        "properties": {"pixelSize": w["pixelSize"]},
        "fields":"pixelSize"}} for w in widths]}
    ws.spreadsheet.batch_update(body)

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

    st.markdown("""
        <div class="care-accent">
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

    with st.spinner("Indexing exemplars (per question)..."):
        ex_index, attr_index, global_pack, by_q, qtexts = build_exemplar_index(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found.")
        return

    # --- Section: Raw fetched dataset ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Raw dataset")
    st.caption(f"Rows: {len(df):,}  •  Columns: {len(df.columns):,}")
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Section: Scored table ---
    with st.spinner("Scoring (+ AI detection)..."):
        scored = score_dataframe(df, mapping, ex_index, attr_index, global_pack, by_q, qtexts)

    st.success("✅ Scoring complete.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Scored table")
    st.caption(
        "Date → Duration → Care_Staff, then source columns (excluded set removed), "
        "per-question scores & rubrics, attribute averages, Overall score, Overall Rank, "
        "and AI flag as the last column."
    )
    st.dataframe(scored, use_container_width=True, height=520)
    st.markdown('</div>', unsafe_allow_html=True)

    # Export
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
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="Leadership_Scoring.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Optional push to Google Sheets
    if AUTO_PUSH:
        ok, msg = upload_df_to_gsheets(scored)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    else:
        st.info("Google Sheets push is disabled (AUTO_PUSH=false).")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
