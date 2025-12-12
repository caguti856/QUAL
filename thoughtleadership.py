# file: thoughtleadership_meaning_only.py
# Purpose: Streamlit app to fetch Kobo submissions and score open-ended responses (0–3)
#          using ONLY meaning similarity to exemplar ANSWERS ("text" in JSONL).
#
# Design goals (per your request):
# - Length-neutral: long/short answers are treated the same (no word/char rules).
# - Score ONLY by meaning similarity to exemplar answers for the mapped question_id.
# - Uses multiple exemplars (top-K) to produce a score distribution (0..3).
# - Uses "consensus among exemplars" to FLAG Needs_Review when meanings disagree,
#   but does NOT downgrade scores based on length/verbosity.
# - Fully automated: no sidebar tuning.
#
# Dependencies: streamlit, pandas, numpy, requests, rapidfuzz, sentence-transformers, gspread, google-auth
#
# Run: streamlit run thoughtleadership_meaning_only.py

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

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
        .main .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1200px; }
        h1, h2, h3 { font-family: "Segoe UI", system-ui, sans-serif; color: var(--text-main); }
        p, span, label { color: var(--text-muted); }
        .app-header-card {
            background: radial-gradient(circle at top left, rgba(242,106,33,0.15), #ffffff);
            border-radius: 1.25rem;
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.10);
            margin-bottom: 1.0rem;
        }
        .pill {
            display: inline-block; font-size: 0.75rem; padding: 0.15rem 0.7rem; border-radius: 999px;
            background: rgba(242,106,33,0.10); border: 1px solid rgba(242,106,33,0.6);
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
MAPPING_PATH = DATASETS_DIR / "mapping1.csv"
EXEMPLARS_PATH = DATASETS_DIR / "thought_leadership.cleaned.jsonl"

# Local/dev fallbacks:
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
# SCORING CONFIG (AUTOMATED; no sidebar)
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

# Similarity voting
KNN_K = int(st.secrets.get("KNN_K", 18))                 # number of exemplar answers to vote
TEMP = float(st.secrets.get("KNN_TEMP", 0.10))          # softmax temperature (meaning smoothing)

# Mix if question pack is sparse (still meaning-based)
WQ, WA, WG = 0.80, 0.15, 0.05

# Consensus (meaning agreement) - for Needs_Review flag ONLY
CLUSTER_SIM = float(st.secrets.get("CLUSTER_SIM", 0.80))  # exemplar↔exemplar meaning agreement threshold
CONSENSUS_MIN_SHARE = float(st.secrets.get("CONSENSUS_MIN_SHARE", 0.50))
CONSENSUS_MIN_PURITY = float(st.secrets.get("CONSENSUS_MIN_PURITY", 0.60))

# Fast duplicate reuse (meaning identical-ish)
DUP_SIM = float(st.secrets.get("DUP_SIM", 0.95))

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
def clean(s) -> str:
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


# =============================================================================
# EMBEDDINGS (batched + cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_embedder():
    # If your hosting environment blocks downloads, pre-cache/bundle this model.
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
    normed = []
    seen = set()
    for t in texts:
        t = clean(t)
        if not t or t in seen:
            continue
        seen.add(t)
        normed.append(t)

    missing = [t for t in normed if t not in _EMB_CACHE]
    if not missing:
        return
    pack = _embed_texts_cached(tuple(missing))
    _EMB_CACHE.update(pack)

def emb_of(text: str) -> Optional[np.ndarray]:
    t = clean(text)
    return _EMB_CACHE.get(t)


# =============================================================================
# EXEMPLAR PACKS
# =============================================================================
@dataclass
class ScoreDist:
    dist: np.ndarray          # (4,)
    expected: float           # 0..3
    pred: int                 # 0..3
    conf: float               # max(dist)
    max_sim: float            # best exemplar similarity
    consensus_share: float    # largest meaning cluster share
    purity: float             # dominant score share within largest cluster
    method: str               # metadata label

@dataclass
class ExemplarPack:
    vecs: np.ndarray          # (n, d) normalized
    scores: np.ndarray        # (n,) ints 0..3
    texts: List[str]          # (n,) exemplar answers
    counts: np.ndarray        # (4,) number per class


def read_jsonl_path(path: Path) -> List[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith(","):
                s = s.lstrip(",").strip()
            out.append(json.loads(s))
    return out


def _build_pack(texts: List[str], scores: List[int]) -> ExemplarPack:
    # De-dup exact (text,score) pairs
    seen = set()
    tt, ss = [], []
    for t, s in zip(texts, scores):
        t = clean(t)
        if not t:
            continue
        try:
            s = int(s)
        except Exception:
            s = int(float(s))
        key = (t, s)
        if key in seen:
            continue
        seen.add(key)
        tt.append(t)
        ss.append(s)

    embed_many(tt)
    vecs = [emb_of(t) for t in tt]
    keep = [(t, s, v) for t, s, v in zip(tt, ss, vecs) if v is not None]
    if not keep:
        return ExemplarPack(vecs=np.zeros((0, 384), dtype=np.float32),
                            scores=np.zeros((0,), dtype=np.int32),
                            texts=[],
                            counts=np.zeros(4, dtype=np.int32))

    texts2 = [t for t, _, _ in keep]
    scores2 = np.array([int(s) for _, s, _ in keep], dtype=np.int32)
    mat = np.vstack([v for _, _, v in keep]).astype(np.float32)

    counts = np.zeros(4, dtype=np.int32)
    for s in range(4):
        counts[s] = int(np.sum(scores2 == s))

    return ExemplarPack(vecs=mat, scores=scores2, texts=texts2, counts=counts)


def build_exemplar_packs(exemplars: List[dict]):
    by_qid: Dict[str, dict] = {}
    by_attr: Dict[str, dict] = {}

    all_texts: List[str] = []

    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        qtext = clean(e.get("question_text", ""))
        txt = clean(e.get("text", ""))
        attr = clean(e.get("attribute", ""))
        if not qid or not txt:
            continue
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0

        q = by_qid.setdefault(qid, {"texts": [], "scores": [], "question_text": qtext, "attribute": attr})
        q["texts"].append(txt)
        q["scores"].append(sc)

        if attr:
            a = by_attr.setdefault(attr, {"texts": [], "scores": []})
            a["texts"].append(txt)
            a["scores"].append(sc)

        all_texts.append(txt)

    # Embed all exemplar texts once (batch)
    embed_many(list(set(all_texts)))

    q_packs = {qid: _build_pack(v["texts"], v["scores"]) for qid, v in by_qid.items()}
    a_packs = {attr: _build_pack(v["texts"], v["scores"]) for attr, v in by_attr.items()}

    # Global pack
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

    # Keep question texts for mapping sanity checks / debugging if needed
    q_texts = [clean(v.get("question_text", "")) for v in by_qid.values() if clean(v.get("question_text", ""))]
    seen = set()
    q_texts = [x for x in q_texts if not (x in seen or seen.add(x))]

    return q_packs, a_packs, g_pack, by_qid, q_texts


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)


def _consensus_metrics(selected_vecs: np.ndarray, selected_scores: np.ndarray, sim_thr: float) -> Tuple[float, float]:
    """
    Consensus among selected EXEMPLARS (not the answer):
    - Build edges between exemplars if cosine sim >= sim_thr
    - Largest component share = consensus_share
    - Purity = dominant score proportion in that component
    """
    n = selected_vecs.shape[0]
    if n <= 1:
        return 1.0, 1.0

    sims = selected_vecs @ selected_vecs.T
    adj = sims >= float(sim_thr)
    np.fill_diagonal(adj, False)

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

    xs, ys = np.where(adj)
    for a, b in zip(xs.tolist(), ys.tolist()):
        union(a, b)

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    uniq, cnts = np.unique(roots, return_counts=True)
    largest_root = int(uniq[int(np.argmax(cnts))])
    comp = np.where(roots == largest_root)[0]

    consensus_share = float(len(comp) / n)

    comp_scores = selected_scores[comp].astype(int)
    vals, cts = np.unique(comp_scores, return_counts=True)
    purity = float(np.max(cts) / len(comp)) if len(comp) else 0.0
    return consensus_share, purity


def score_against_pack(pack: ExemplarPack, ans_vec: np.ndarray) -> Optional[ScoreDist]:
    if pack is None or pack.vecs.size == 0 or ans_vec is None:
        return None

    sims = pack.vecs @ ans_vec  # cosine similarity (normalized embeddings)
    if sims.size == 0:
        return None

    k = max(1, min(int(KNN_K), sims.size))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    top_sims = sims[idx]
    top_scores = pack.scores[idx].astype(int)
    top_vecs = pack.vecs[idx]

    w = _softmax((top_sims - float(top_sims.max())) / max(1e-6, float(TEMP)))
    dist = np.zeros(4, dtype=np.float32)
    for s, wi in zip(top_scores, w):
        if 0 <= int(s) <= 3:
            dist[int(s)] += float(wi)
    dist = dist / (dist.sum() + 1e-9)

    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())
    max_sim = float(top_sims.max())

    consensus_share, purity = _consensus_metrics(top_vecs, top_scores, CLUSTER_SIM)

    return ScoreDist(
        dist=dist,
        expected=expected,
        pred=pred,
        conf=conf,
        max_sim=max_sim,
        consensus_share=consensus_share,
        purity=purity,
        method="meaning_knn_vote",
    )


def _mix(dq: Optional[ScoreDist], da: Optional[ScoreDist], dg: Optional[ScoreDist]) -> Optional[ScoreDist]:
    items = []
    weights = []
    for d, w in [(dq, WQ), (da, WA), (dg, WG)]:
        if d is not None and w > 0:
            items.append(d)
            weights.append(float(w))
    if not items:
        return None

    ws = np.array(weights, dtype=np.float32)
    ws = ws / (ws.sum() + 1e-9)
    mat = np.vstack([d.dist for d in items]).astype(np.float32)
    dist = (mat * ws[:, None]).sum(axis=0)
    dist = dist / (dist.sum() + 1e-9)

    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())
    max_sim = float(max(d.max_sim for d in items))

    # Keep the worst (most cautious) consensus stats to decide Needs_Review
    consensus_share = float(min(d.consensus_share for d in items))
    purity = float(min(d.purity for d in items))

    return ScoreDist(
        dist=dist,
        expected=expected,
        pred=pred,
        conf=conf,
        max_sim=max_sim,
        consensus_share=consensus_share,
        purity=purity,
        method="meaning_mix(q/a/g)",
    )


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
# DATA LOADERS / COLUMN RESOLUTION
# =============================================================================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "prompt_hint" not in df.columns and "column" in df.columns:
        df = df.rename(columns={"column": "prompt_hint"})
    if "question_id" not in df.columns:
        raise ValueError("mapping.csv must include column 'question_id'")
    if "attribute" not in df.columns:
        raise ValueError("mapping.csv must include column 'attribute'")
    return df


_QID_PREFIX_TO_SECTION = {
    "LAV": "A1", "II": "A2", "EP": "A3", "CFC": "A4", "FTD": "A5", "LDA": "A6", "RDM": "A7",
}

def _score_kobo_header(col: str, token: str) -> int:
    col = clean(col).lower()
    token = clean(token).lower()
    if not col or not token:
        return 0
    if token in col:
        return 100
    return int(fuzz.token_set_ratio(token, col))

def resolve_kobo_column_for_mapping(df_cols: List[str], qid: str, prompt_hint: str) -> Optional[str]:
    qid = (qid or "").strip()
    if not qid:
        return None

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
# EXPORTS / SHEETS
# =============================================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return bio.getvalue()

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

def upload_df_to_gsheets(df: pd.DataFrame) -> Tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        header = df.columns.astype(str).tolist()
        values = df.astype(object).where(pd.notna(df), "").values.tolist()
        ws.clear()
        ws.spreadsheet.values_batch_update(
            body={"valueInputOption": "USER_ENTERED", "data": [{"range": f"'{ws.title}'!A1", "values": [header] + values}]}
        )
        return True, f"✅ Wrote {len(values)} rows × {len(header)} cols to '{ws.title}'."
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"


# =============================================================================
# MAIN SCORER (meaning-only)
# =============================================================================
def score_dataframe(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    q_packs: Dict[str, ExemplarPack],
    a_packs: Dict[str, ExemplarPack],
    g_pack: ExemplarPack,
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

    # Resolve Kobo columns per question id
    all_mapping = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute", "")) in ORDERED_ATTRS]
    resolved_for_qid: Dict[str, str] = {}
    for r in all_mapping:
        qid = clean(r.get("question_id", ""))
        qhint = r.get("prompt_hint", "") or r.get("column", "")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit:
            resolved_for_qid[qid] = hit

    # Pre-embed all distinct answers (batch)
    distinct_answers: set = set()
    for _, row in df.iterrows():
        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a:
                    distinct_answers.add(a)
    embed_many(list(distinct_answers))

    # Duplicate reuse by question id
    dup_bank: Dict[str, List[Tuple[np.ndarray, int]]] = {}

    out_rows = []
    for _, resp in df.iterrows():
        row = {}
        row["Date"] = clean(resp.get(date_col, ""))

        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = clean(resp.get(who_col, "")) if who_col else ""

        # Pass-through metadata
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date", "Care_Staff"):
                continue
            row[c] = resp.get(c, "")

        per_attr: Dict[str, List[int]] = {}
        needs_review = False

        # Score each question
        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            attr = clean(r.get("attribute", ""))
            col = resolved_for_qid.get(qid)
            if not col or col not in df.columns:
                continue

            ans = clean(resp.get(col, ""))
            if not ans:
                continue

            ans_vec = emb_of(ans)
            if ans_vec is None:
                continue

            # Fast duplicate reuse (meaning-level)
            reused = False
            best_dup_sc = None
            best_dup_sim = -1.0
            for v2, sc2 in dup_bank.get(qid, []):
                sim = float(np.dot(ans_vec, v2))
                if sim > best_dup_sim:
                    best_dup_sim, best_dup_sc = sim, sc2
            if best_dup_sc is not None and best_dup_sim >= DUP_SIM:
                score = int(best_dup_sc)
                conf = 1.0
                consensus_share = 1.0
                purity = 1.0
                reused = True
            else:
                dq = score_against_pack(q_packs.get(qid), ans_vec)
                da = score_against_pack(a_packs.get(attr), ans_vec) if attr else None
                dg = score_against_pack(g_pack, ans_vec)

                d = _mix(dq, da, dg) if (dq or da or dg) else None
                if d is None:
                    # If no exemplars at all, default to Compliant and flag
                    score, conf, consensus_share, purity = 1, 0.25, 0.0, 0.0
                    needs_review = True
                else:
                    score = int(d.pred)
                    conf = float(d.conf)
                    consensus_share = float(d.consensus_share)
                    purity = float(d.purity)

                    # Needs_Review is ONLY about meaning disagreement among top exemplars (not length)
                    if (consensus_share < CONSENSUS_MIN_SHARE) or (purity < CONSENSUS_MIN_PURITY):
                        needs_review = True

            # Cache for duplicates
            if not reused:
                bank = dup_bank.setdefault(qid, [])
                if len(bank) < 400:
                    bank.append((ans_vec, int(score)))

            # Output columns: attribute question numbers
            qn = None
            if "_Q" in qid:
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None

            if qn in (1, 2, 3, 4):
                row[f"{attr}_Qn{qn}"] = int(score)
                row[f"{attr}_Rubric_Qn{qn}"] = BANDS[int(score)]
                row[f"{attr}_Qn{qn}_Conf"] = round(float(conf), 3)
                row[f"{attr}_Qn{qn}_Consensus"] = round(float(consensus_share), 3)
                row[f"{attr}_Qn{qn}_Purity"] = round(float(purity), 3)
                per_attr.setdefault(attr, []).append(int(score))

        # Fill blanks
        for attr in ORDERED_ATTRS:
            for qn in (1, 2, 3, 4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")
                row.setdefault(f"{attr}_Qn{qn}_Conf", "")
                row.setdefault(f"{attr}_Qn{qn}_Consensus", "")
                row.setdefault(f"{attr}_Qn{qn}_Purity", "")

        # Attribute averages + overall
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
        row["Needs_Review"] = bool(needs_review)
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # Column order
    ordered = [c for c in ["Date", "Care_Staff"] if c in res.columns]
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date", "Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    # Per-question columns
    mid = []
    for attr in ORDERED_ATTRS:
        for qn in (1, 2, 3, 4):
            mid += [
                f"{attr}_Qn{qn}",
                f"{attr}_Rubric_Qn{qn}",
                f"{attr}_Qn{qn}_Conf",
                f"{attr}_Qn{qn}_Consensus",
                f"{attr}_Qn{qn}_Purity",
            ]
    ordered += [c for c in mid if c in res.columns]

    # Per-attribute columns
    for attr in ORDERED_ATTRS:
        ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]

    ordered += ["Overall Total (0–21)", "Overall Rank", "Needs_Review"]

    return res.reindex(columns=[c for c in ordered if c in res.columns])


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    inject_css()
    st.markdown(
        """
        <div class="app-header-card">
            <div class="pill">Thought Leadership • Meaning-only Scoring</div>
            <h1>Thought Leadership</h1>
            <p>
                Scores each response by meaning similarity to exemplar <b>answers</b> (JSONL <code>text</code>) for the mapped question_id.
                No length-based scoring rules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    with st.spinner("Building exemplar banks (meaning-only)..."):
        q_packs, a_packs, g_pack, _, _ = build_exemplar_packs(exemplars)

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

    with st.spinner("Scoring (meaning-only)..."):
        scored = score_dataframe(df, mapping, q_packs, a_packs, g_pack)

    st.success("✅ Scoring complete.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Scored table")
    st.caption("Scores come from meaning similarity to exemplar answers. Needs_Review flags disagreement among top exemplars.")
    st.dataframe(scored, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⬇️ Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Excel",
            data=to_excel_bytes(scored),
            file_name="Leadership_Scoring_meaning_only.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="Leadership_Scoring_meaning_only.csv",
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
