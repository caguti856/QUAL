#

from __future__ import annotations

import json
import hashlib
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
        [data-testid="stSidebar"] { display: none; } /* no controls */
        .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
        h1, h2, h3 { font-family: "Segoe UI", system-ui, sans-serif; color: var(--text-main); }
        h1 { font-size: 2.1rem; font-weight: 700; }
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
            background: linear-gradient(90deg, #FEF9C3, var(--primary), var(--silver), var(--gold));
            opacity: 0.95;
        }
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

# Local/dev fallbacks
if not MAPPING_PATH.exists():
    alt = Path("/mnt/data/mapping1.csv")
    if alt.exists():
        MAPPING_PATH = alt
if not EXEMPLARS_PATH.exists():
    alt1 = Path("/mnt/data/thought_leadership.cleaned.jsonl")
    alt2 = Path("/mnt/data/thought_leadership_rescored.jsonl")
    alt3 = Path("/mnt/data/thought_leadership.jsonl")
    for a in (alt1, alt2, alt3):
        if a.exists():
            EXEMPLARS_PATH = a
            break


# =============================================================================
# CONFIG (automatic; no UI)
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

# Embeddings
MODEL_NAME = st.secrets.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# Similarity retrieval
TOP_CANDIDATES = int(st.secrets.get("TOP_CANDIDATES", 80))  # retrieve this many, then cluster
VOTE_K = int(st.secrets.get("VOTE_K", 20))                  # max exemplars used for voting (within cluster)
SOFTMAX_TEMP = float(st.secrets.get("SOFTMAX_TEMP", 0.10))  # smaller = sharper vote

# Consensus clustering among candidates (meaning agreement)
CLUSTER_SIM = float(st.secrets.get("CLUSTER_SIM", 0.82))    # how similar exemplars must be to be "same meaning"
MIN_CLUSTER_SIZE = int(st.secrets.get("MIN_CLUSTER_SIZE", 3))

# Pack mixing (question > attribute > global)
WQ, WA, WG = float(st.secrets.get("WQ", 0.75)), float(st.secrets.get("WA", 0.20)), float(st.secrets.get("WG", 0.05))

# Light centroid smoothing (stability when exemplars sparse)
CENTROID_ALPHA = float(st.secrets.get("CENTROID_ALPHA", 0.75))  # 0..1 (higher = rely more on vote)

# Off-topic flag (meaning-only; does not cap score)
OFFTOPIC_FLAG_QSIM = float(st.secrets.get("OFFTOPIC_FLAG_QSIM", 0.06))

# Duplicate reuse (speed)
DUP_SIM = float(st.secrets.get("DUP_SIM", 0.94))

# Debug panel (does not change scores)
SHOW_MATCHES = bool(st.secrets.get("SHOW_MATCHES", True))
MATCHES_TO_SHOW = int(st.secrets.get("MATCHES_TO_SHOW", 6))

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

_WORD_RX = re.compile(r"\w+")


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


def qa_overlap(ans: str, qtext: str) -> float:
    ans_s = clean(ans).lower()
    q_s = clean(qtext).lower()
    at = set(re.findall(r"\w+", ans_s))
    qt = set(re.findall(r"\w+", q_s))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0


# =============================================================================
# EMBEDDINGS (memory + disk cache)
# =============================================================================
EMB_DISK_CACHE_DIR = Path(st.secrets.get("EMB_DISK_CACHE_DIR", ".emb_cache_tl"))
EMB_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(MODEL_NAME)

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

def _emb_cache_path(text: str) -> Path:
    h = hashlib.sha1(clean(text).encode("utf-8", errors="ignore")).hexdigest()
    return EMB_DISK_CACHE_DIR / f"{h}.npy"

def embed_many(texts: List[str]) -> None:
    # normalize + unique
    normed, seen = [], set()
    for t in texts:
        t = clean(t)
        if not t or t in seen:
            continue
        seen.add(t)
        normed.append(t)

    # load from disk cache first
    missing = [t for t in normed if t not in _EMB_CACHE]
    if not missing:
        return

    still_missing = []
    for t in missing:
        p = _emb_cache_path(t)
        if p.exists():
            try:
                _EMB_CACHE[t] = np.load(p)
            except Exception:
                still_missing.append(t)
        else:
            still_missing.append(t)

    if not still_missing:
        return

    pack = _embed_texts_cached(tuple(still_missing))
    _EMB_CACHE.update(pack)
    for t in still_missing:
        try:
            np.save(_emb_cache_path(t), _EMB_CACHE[t])
        except Exception:
            pass

def emb_of(text: str) -> Optional[np.ndarray]:
    t = clean(text)
    return _EMB_CACHE.get(t, None)


# =============================================================================
# EXEMPLAR PACKS
# =============================================================================
@dataclass
class ScoreDist:
    dist: np.ndarray            # (4,)
    expected: float             # 0..3
    pred: int                   # 0..3
    conf: float                 # 0..1
    max_sim: float              # similarity of best exemplar in chosen cluster
    cluster_size: int           # exemplars used after clustering
    method: str                 # label

@dataclass
class ExemplarPack:
    vecs: np.ndarray            # (n, d)
    scores: np.ndarray          # (n,)
    texts: List[str]            # (n,)
    centroids: np.ndarray       # (4, d)
    counts: np.ndarray          # (4,)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)

def _build_pack(texts: List[str], scores: List[int]) -> ExemplarPack:
    # de-dup exact (text, score)
    seen = set()
    tt, ss = [], []
    for t, s in zip(texts, scores):
        t = clean(t)
        if not t:
            continue
        try:
            s = int(s)
        except Exception:
            try:
                s = int(float(s))
            except Exception:
                s = 0
        key = (t, int(s))
        if key in seen:
            continue
        seen.add(key)
        tt.append(t)
        ss.append(int(s))

    embed_many(list(set(tt)))

    vecs, keep_t, keep_s = [], [], []
    for t, s in zip(tt, ss):
        v = emb_of(t)
        if v is None:
            continue
        vecs.append(v)
        keep_t.append(t)
        keep_s.append(int(s))

    if vecs:
        mat = np.vstack(vecs).astype(np.float32)
    else:
        mat = np.zeros((0, 384), dtype=np.float32)

    scores_arr = np.array(keep_s, dtype=np.int32)

    # per-score centroids
    d = mat.shape[1] if mat.size else 384
    centroids = np.zeros((4, d), dtype=np.float32)
    counts = np.zeros(4, dtype=np.int32)
    for s in range(4):
        idx = np.where(scores_arr == s)[0]
        counts[s] = len(idx)
        if len(idx) >= 1:
            c = mat[idx].mean(axis=0)
            n = np.linalg.norm(c) + 1e-9
            centroids[s] = (c / n).astype(np.float32)
        else:
            centroids[s] = np.zeros((d,), dtype=np.float32)

    return ExemplarPack(vecs=mat, scores=scores_arr, texts=keep_t, centroids=centroids, counts=counts)

def build_exemplar_packs(exemplars: List[dict]):
    by_qkey: Dict[str, dict] = {}
    by_attr: Dict[str, dict] = {}
    question_texts: List[str] = []
    all_texts, all_qtexts = [], []

    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        qtext = clean(e.get("question_text", ""))
        txt = clean(e.get("text", ""))              # <-- EXEMPLAR ANSWER TEXT
        attr = clean(e.get("attribute", ""))
        try:
            sc = int(e.get("score", 0))
        except Exception:
            try:
                sc = int(float(e.get("score", 0)))
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

    embed_many(list(set(all_texts + all_qtexts)))

    q_packs = {k: _build_pack(v["texts"], v["scores"]) for k, v in by_qkey.items()}
    a_packs = {a: _build_pack(v["texts"], v["scores"]) for a, v in by_attr.items()}
    g_pack = _build_pack(all_texts, [int(clean(e.get("score", 0)) or 0) for e in exemplars if clean(e.get("text", ""))])

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


# =============================================================================
# ROBUST EXEMPLAR READER
# =============================================================================
def read_exemplars(path: Path) -> List[dict]:
    """
    Accepts:
      - proper JSONL (one object per line)
      - a JSON array of objects
      - concatenated JSON objects (with optional commas/newlines)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Exemplars file not found: {path}")

    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # Try JSON array first
    if raw[0] == "[":
        data = json.loads(raw)
        return [x for x in data if isinstance(x, dict)]

    # Otherwise parse sequential JSON objects
    out: List[dict] = []
    dec = json.JSONDecoder()
    i, n = 0, len(raw)
    while i < n:
        # skip whitespace / newlines / commas
        while i < n and raw[i] in " \r\n\t,":
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(raw, i)
        i = j
        if isinstance(obj, dict):
            out.append(obj)

    return out


# =============================================================================
# CONSENSUS CLUSTERING + VOTING (meaning-only)
# =============================================================================
def _union_find_components(adj: np.ndarray) -> List[np.ndarray]:
    n = adj.shape[0]
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
        if a < b:
            union(a, b)

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    comps = []
    for r in np.unique(roots):
        comps.append(np.where(roots == r)[0])
    return comps

def score_against_pack_meaning(pack: ExemplarPack, ans_vec: np.ndarray, return_matches: bool = False):
    """
    Core idea:
      1) Similarity search: find top-N exemplar ANSWERS closest to the response.
      2) Meaning consensus: among those top-N, keep the largest/strongest cluster
         of exemplars that agree with each other (not just with the response).
      3) Score = weighted vote of exemplar scores inside that cluster.
      4) Light centroid smoothing for stability.
    """
    if pack is None or pack.vecs.size == 0 or ans_vec is None:
        return None if not return_matches else (None, [])

    sims = pack.vecs @ ans_vec
    if sims.size == 0:
        return None if not return_matches else (None, [])

    N = max(VOTE_K, min(int(TOP_CANDIDATES), sims.size))
    idx = np.argpartition(-sims, N - 1)[:N]
    idx = idx[np.argsort(-sims[idx])]

    cand_vecs = pack.vecs[idx]
    cand_sims = sims[idx].astype(np.float32)
    cand_scores = pack.scores[idx].astype(np.int32)

    # Build exemplar-to-exemplar adjacency for consensus
    sim_mat = cand_vecs @ cand_vecs.T
    adj = sim_mat >= float(CLUSTER_SIM)
    np.fill_diagonal(adj, False)

    comps = _union_find_components(adj) if cand_vecs.shape[0] >= 2 else [np.array([0], dtype=np.int64)]

    # Choose the "best" meaning cluster:
    # highest total similarity to the answer (and prefer larger clusters).
    best = None
    best_val = -1e9
    for c in comps:
        size = int(c.size)
        if size < MIN_CLUSTER_SIZE and len(comps) > 1:
            continue
        sim_sum = float(cand_sims[c].sum())
        val = sim_sum + 0.05 * size  # small size bonus, but similarity dominates
        if val > best_val:
            best_val = val
            best = c

    if best is None:
        best = np.arange(min(VOTE_K, cand_sims.size), dtype=np.int64)

    # Within cluster, keep at most VOTE_K (highest similarity to answer)
    local = best[np.argsort(-cand_sims[best])]
    local = local[: max(1, min(int(VOTE_K), local.size))]

    top_sims = cand_sims[local]
    top_scores = cand_scores[local]
    top_texts = [pack.texts[int(idx[int(i)])] for i in local.tolist()] if return_matches else None

    # Weighted vote distribution by score label
    # IMPORTANT: weights depend ONLY on similarity to the response (meaning), not length.
    w = _softmax((top_sims - float(top_sims.max())) / max(1e-6, float(SOFTMAX_TEMP)))
    vote = np.zeros(4, dtype=np.float32)
    for s, wi in zip(top_scores.tolist(), w.tolist()):
        if 0 <= int(s) <= 3:
            vote[int(s)] += float(wi)

    # Centroid smoothing (optional)
    cent = pack.centroids @ ans_vec
    cent_mask = (pack.counts >= 1).astype(np.float32)
    cent = cent * cent_mask + (-10.0) * (1.0 - cent_mask)
    cent_dist = _softmax(cent / max(1e-6, float(SOFTMAX_TEMP)))

    alpha = float(np.clip(CENTROID_ALPHA, 0.0, 1.0))
    dist = alpha * vote + (1.0 - alpha) * cent_dist
    dist = dist / (dist.sum() + 1e-9)

    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())

    sd = ScoreDist(
        dist=dist,
        expected=expected,
        pred=pred,
        conf=conf,
        max_sim=float(top_sims.max()) if top_sims.size else float(cand_sims.max()),
        cluster_size=int(local.size),
        method="meaning_cluster_vote+centroid",
    )

    if not return_matches:
        return sd

    matches = []
    for t, s, sim in zip(top_texts[:MATCHES_TO_SHOW], top_scores[:MATCHES_TO_SHOW], top_sims[:MATCHES_TO_SHOW]):
        matches.append({"score": int(s), "sim": float(sim), "text": t[:260]})
    return sd, matches

def _mix(d1: Optional[ScoreDist], d2: Optional[ScoreDist], d3: Optional[ScoreDist]) -> Optional[ScoreDist]:
    present = [(d1, WQ), (d2, WA), (d3, WG)]
    present = [(d, float(w)) for d, w in present if d is not None and float(w) > 0]
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
    cluster_size = int(max(d.cluster_size for d, _ in present))
    return ScoreDist(dist=dist, expected=expected, pred=pred, conf=conf, max_sim=max_sim, cluster_size=cluster_size, method="mix(q,a,g)")


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
    if "prompt_hint" not in df.columns and "column" in df.columns:
        df = df.rename(columns={"column": "prompt_hint"})
    for req in ("question_id", "attribute"):
        if req not in df.columns:
            raise ValueError(f"mapping.csv must include column '{req}'")
    return df


# =============================================================================
# KOBO COLUMN RESOLUTION
# =============================================================================
_QID_PREFIX_TO_SECTION = {"LAV": "A1", "II": "A2", "EP": "A3", "CFC": "A4", "FTD": "A5", "LDA": "A6", "RDM": "A7"}

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
# SCORING (meaning only)
# =============================================================================
def score_dataframe(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    q_packs: Dict[str, ExemplarPack],
    a_packs: Dict[str, ExemplarPack],
    g_pack: ExemplarPack,
    by_qkey: Dict[str, dict],
    question_texts: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:

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

    # Date parsing
    if date_col in df.columns:
        date_clean = df[date_col].astype(str).str.strip().str.lstrip(",")
        dt_series = pd.to_datetime(date_clean, errors="coerce")
    else:
        dt_series = pd.Series([pd.NaT] * n_rows)

    # duration
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

    # Filter mapping to known attributes
    all_mapping = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute", "")) in ORDERED_ATTRS]

    # Resolve Kobo columns per question id
    resolved_for_qid: Dict[str, str] = {}
    for r in all_mapping:
        qid = clean(r.get("question_id", ""))
        qhint = r.get("prompt_hint", "") or r.get("column", "")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit:
            resolved_for_qid[qid] = hit

    # Pre-embed all distinct answers + question refs (for off-topic flag only)
    distinct_answers = set()
    qrefs = set()
    for _, row in df.iterrows():
        for r in all_mapping:
            qid = clean(r.get("question_id", ""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a:
                    distinct_answers.add(a)

    for r in all_mapping:
        qid = clean(r.get("question_id", ""))
        qhint = clean(r.get("prompt_hint", "") or r.get("column", ""))
        qkey = resolve_qkey(q_packs, by_qkey, question_texts, qid, qhint)
        if qkey:
            qt = clean((by_qkey.get(qkey, {}) or {}).get("question_text", ""))
            if qt:
                qrefs.add(qt)
        if qhint:
            qrefs.add(qhint)

    embed_many(list(distinct_answers | qrefs))

    # caches
    exact_sc_cache: Dict[Tuple[str, str], int] = {}
    dup_bank: Dict[str, List[Tuple[np.ndarray, int]]] = {}

    # optional debug store: key = "rowidx|qid"
    debug_matches: Dict[str, List[dict]] = {}

    out_rows = []
    for i, resp in df.iterrows():
        row = {}
        row["Date"] = (pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_series.iloc[i]) else str(i))
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

            # cached score
            sc = exact_sc_cache.get((qid, ans))
            reused = False

            # duplicate reuse within question (speed only)
            if sc is None:
                best_dup_sc, best_dup_sim = None, -1.0
                for v2, sc2 in dup_bank.get(qid, []):
                    sim = float(np.dot(vec, v2))
                    if sim > best_dup_sim:
                        best_dup_sim, best_dup_sc = sim, sc2
                if best_dup_sc is not None and best_dup_sim >= DUP_SIM:
                    sc = int(best_dup_sc)
                    reused = True

            # meaning-only scoring using exemplar ANSWER TEXT
            qkey = resolve_qkey(q_packs, by_qkey, question_texts, qid, qhint)
            qtext_full = clean((by_qkey.get(qkey, {}) or {}).get("question_text", "")) if qkey else ""
            qref = qtext_full or qhint

            if sc is None:
                # Score vs packs (question, attribute, global) then mix
                ex_q = None
                matches_q = []
                if qkey and qkey in q_packs:
                    if SHOW_MATCHES:
                        ex_q, matches_q = score_against_pack_meaning(q_packs[qkey], vec, return_matches=True)
                    else:
                        ex_q = score_against_pack_meaning(q_packs[qkey], vec, return_matches=False)
                ex_a = score_against_pack_meaning(a_packs[attr], vec) if (attr in a_packs) else None
                ex_g = score_against_pack_meaning(g_pack, vec)

                ex = _mix(ex_q, ex_a, ex_g)
                sc = int(ex.pred) if ex is not None else 1

                # Off-topic flag only (no score change)
                qv = emb_of(qref) if qref else None
                if qv is not None:
                    qsim = float(np.dot(vec, qv))
                    if qsim < OFFTOPIC_FLAG_QSIM:
                        needs_review = True

                # save matches for audit
                if SHOW_MATCHES and matches_q:
                    debug_matches[f"{i}|{qid}"] = matches_q

            # cache scored result
            exact_sc_cache[(qid, ans)] = int(sc)
            if not reused:
                bank = dup_bank.setdefault(qid, [])
                if len(bank) < 400:
                    bank.append((vec, int(sc)))

            # Output columns
            qn = None
            if "_Q" in (qid or ""):
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None

            if qn in (1, 2, 3, 4):
                sk = f"{attr}_Qn{qn}"
                rk = f"{attr}_Rubric_Qn{qn}"
                row[sk] = int(sc)
                row[rk] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

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
        row["Needs_Review"] = bool(needs_review)
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

    ordered += [c for c in ["Overall Total (0–21)", "Overall Rank", "Needs_Review"] if c in res.columns]

    res = res.reindex(columns=[c for c in ordered if c in res.columns])
    return res, debug_matches


# =============================================================================
# EXPORTS / SHEETS
# =============================================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
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

def upload_df_to_gsheets(df: pd.DataFrame) -> Tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        header = df.columns.astype(str).tolist()
        values = df.astype(object).where(pd.notna(df), "").values.tolist()
        ws.clear()
        ws.spreadsheet.values_batch_update(
            body={"valueInputOption": "USER_ENTERED", "data": [{"range": f"'{ws.title}'!A1", "values": [header] + values}]}
        )
        try:
            ws.freeze(rows=1)
        except Exception:
            pass
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
            <div class="pill">Thought Leadership • Meaning-only Scoring</div>
            <h1>Thought Leadership</h1>
            <p style="color:#6b7280;margin:0;">
                Scores each response by comparing it to exemplar <b>answer texts</b> and voting inside a consensus meaning cluster.
                Question text is only used to pick the right exemplar set and (optionally) flag off-topic.
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
        exemplars = read_exemplars(EXEMPLARS_PATH)
        if not exemplars:
            st.error("Exemplars file is empty (or could not be parsed).")
            return
    except Exception as e:
        st.error(f"Failed to read exemplars: {e}")
        return

    with st.spinner("Building exemplar banks..."):
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

    with st.spinner("Scoring (meaning-only)…"):
        scored, matches = score_dataframe(df, mapping, q_packs, a_packs, g_pack, by_q, qtexts)

    st.success("✅ Scoring complete.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Scored table")
    st.dataframe(scored, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if SHOW_MATCHES and matches:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🔎 Why a score was chosen (top matching exemplar answers)")
        st.caption("This is the proof that the scorer is comparing your response to the JSONL 'text' field.")
        key = st.selectbox("Pick a row/question to inspect", list(matches.keys()))
        st.write(matches.get(key, []))
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
            data=scored.to_csv(index=False).encode("utf-8"),
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
