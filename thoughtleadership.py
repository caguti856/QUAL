
from __future__ import annotations

import hashlib
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
from sentence_transformers import SentenceTransformer, CrossEncoder


# =============================================================================
# UI / STYLING (minimal)
# =============================================================================
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
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1, h2, h3 {
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-main);
        }
        h1 { font-size: 2.1rem; font-weight: 700; }
        h2 { margin-top: 1.5rem; font-size: 1.3rem; }
        p, span, label { color: var(--text-muted); }

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
        .app-header-subtitle { font-size: 0.9rem; color: var(--text-muted); }

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

        .section-card {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border-subtle);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.04);
        }

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
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)


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
    alt2 = Path("/mnt/data/thought_leadership.jsonl")
    if alt1.exists():
        EXEMPLARS_PATH = alt1
    elif alt2.exists():
        EXEMPLARS_PATH = alt2


# =============================================================================
# SCORING CONSTANTS (AUTO)
# =============================================================================
BANDS = {0: "Counterproductive", 1: "Compliant", 2: "Strategic", 3: "Transformative"}
ORDERED_ATTRS = [
    "Locally Anchored Visioning",
    "Innovation and Insight",
    "Execution Planning",
    "Cross-Functional Collaboration",
    "Follow-Through Discipline",
    "Learning-Driven Adjustment",
    "Result-Oriented Decision-Making",
]

# Retrieval + rerank
TOPN_RETRIEVE = 70          # retrieve by embeddings (fast)
TOPM_RERANK = 40            # rerank best M from retrieved (accurate)
SEED_TRIES = 6              # cluster seeds tried from reranked list
CLUSTER_SIM = 0.78          # cosine sim threshold among exemplars for "same meaning"
MIN_CLUSTER = 4             # minimum exemplars in cluster to trust cluster vote
RERANK_TEMP = 0.35          # softmax temperature over reranker scores (lower = sharper)

# Off-topic (very conservative)
OFFTOPIC_QSIM = 0.06        # only if answer is truly unrelated
OFFTOPIC_CAP = 2            # if off-topic, cap at <=2 (still not punitive)

# Model choices (fast + strong)
BI_ENCODER_NAME = st.secrets.get("BI_ENCODER_NAME", "intfloat/e5-base-v2")
CROSS_ENCODER_NAME = st.secrets.get("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Caches (disk)
CACHE_DIR = Path(st.secrets.get("TL_CACHE_DIR", ".tl_cache"))
EMB_DIR = CACHE_DIR / "emb"
RERANK_DIR = CACHE_DIR / "rerank"
PACK_DIR = CACHE_DIR / "packs"
for d in (EMB_DIR, RERANK_DIR, PACK_DIR):
    d.mkdir(parents=True, exist_ok=True)

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
_SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+|\n+")


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

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _softmax_temp(x: np.ndarray, temp: float) -> np.ndarray:
    x = x.astype(np.float32)
    t = max(1e-6, float(temp))
    x = x / t
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)

def split_sentences(text: str) -> List[str]:
    t = clean(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RX.split(t) if p and p.strip()]
    return parts


# =============================================================================
# MODELS (cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_biencoder() -> SentenceTransformer:
    return SentenceTransformer(BI_ENCODER_NAME)

@st.cache_resource(show_spinner=False)
def get_crossencoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_NAME)

def _is_e5_model() -> bool:
    return "e5" in BI_ENCODER_NAME.lower()

def _e5_prefix_query(t: str) -> str:
    return f"query: {t}"

def _e5_prefix_passage(t: str) -> str:
    return f"passage: {t}"

def _emb_cache_path(text: str, kind: str) -> Path:
    # kind: "q" or "p"
    return EMB_DIR / f"{_sha1(kind + '|' + BI_ENCODER_NAME + '|' + clean(text))}.npy"

def embed_texts(texts: List[str], kind: str) -> Dict[str, np.ndarray]:
    """
    kind="q" for queries (answers/questions), kind="p" for passages (exemplars).
    Uses disk cache; batch encodes cache misses.
    """
    out: Dict[str, np.ndarray] = {}
    uniq = []
    seen = set()
    for t in texts:
        t = clean(t)
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)

    missing = []
    for t in uniq:
        p = _emb_cache_path(t, kind)
        if p.exists():
            try:
                out[t] = np.load(p)
            except Exception:
                missing.append(t)
        else:
            missing.append(t)

    if missing:
        model = get_biencoder()
        if _is_e5_model():
            enc_in = [_e5_prefix_query(t) if kind == "q" else _e5_prefix_passage(t) for t in missing]
        else:
            enc_in = missing

        embs = model.encode(
            enc_in,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        ).astype(np.float32)

        for t, v in zip(missing, embs):
            out[t] = v
            try:
                np.save(_emb_cache_path(t, kind), v)
            except Exception:
                pass

    return out


# =============================================================================
# EXEMPLAR PACKS
# =============================================================================
@dataclass
class ExemplarPack:
    vecs: np.ndarray     # (n, d) normalized, exemplar TEXT embeddings (passages)
    scores: np.ndarray   # (n,) int 0..3
    texts: List[str]     # exemplar answers
    qtext: str           # question_text (routing/off-topic only)
    attr: str            # attribute (output column grouping)

@dataclass
class ScoreDist:
    dist: np.ndarray     # (4,)
    expected: float
    pred: int
    conf: float
    max_sim: float
    method: str

def read_jsonl_path(path: Path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith(","):
                s = s.lstrip(",").strip()
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                s2 = re.sub(r",\s*$", "", s)
                out.append(json.loads(s2))
    return out

def _pack_cache_key(qid: str) -> Path:
    return PACK_DIR / f"{_sha1(BI_ENCODER_NAME + '|' + qid)}.npz"

def build_packs(exemplars: List[dict]) -> Dict[str, ExemplarPack]:
    """
    Build one pack per question_id, embedding ONLY exemplar 'text'.
    """
    by_q: Dict[str, dict] = {}
    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        if not qid:
            continue
        txt = clean(e.get("text", ""))
        if not txt:
            continue
        qtext = clean(e.get("question_text", ""))
        attr = clean(e.get("attribute", ""))
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0
        sc = int(np.clip(sc, 0, 3))
        d = by_q.setdefault(qid, {"texts": [], "scores": [], "qtext": qtext, "attr": attr})
        d["texts"].append(txt)
        d["scores"].append(sc)
        if qtext:
            d["qtext"] = qtext
        if attr:
            d["attr"] = attr

    packs: Dict[str, ExemplarPack] = {}
    for qid, d in by_q.items():
        cachep = _pack_cache_key(qid)
        if cachep.exists():
            try:
                npz = np.load(cachep, allow_pickle=True)
                packs[qid] = ExemplarPack(
                    vecs=npz["vecs"].astype(np.float32),
                    scores=npz["scores"].astype(np.int32),
                    texts=list(npz["texts"].tolist()),
                    qtext=str(npz["qtext"].tolist()),
                    attr=str(npz["attr"].tolist()),
                )
                continue
            except Exception:
                pass

        texts = [clean(t) for t in d["texts"] if clean(t)]
        scores = np.array([int(s) for s in d["scores"]], dtype=np.int32)

        emb_map = embed_texts(texts, kind="p")
        vecs = np.vstack([emb_map[t] for t in texts]).astype(np.float32)

        pack = ExemplarPack(vecs=vecs, scores=scores, texts=texts, qtext=d.get("qtext", ""), attr=d.get("attr", ""))
        packs[qid] = pack

        try:
            np.savez_compressed(
                cachep,
                vecs=pack.vecs,
                scores=pack.scores,
                texts=np.array(pack.texts, dtype=object),
                qtext=np.array(pack.qtext, dtype=object),
                attr=np.array(pack.attr, dtype=object),
            )
        except Exception:
            pass

    return packs


# =============================================================================
# RERANK CACHE
# =============================================================================
def _rerank_cache_path(qid: str, ans: str, ex_text: str) -> Path:
    key = _sha1(CROSS_ENCODER_NAME + "|" + qid + "|" + clean(ans) + "|" + clean(ex_text))
    return RERANK_DIR / f"{key}.txt"

def rerank_pairs_cached(qid: str, ans: str, ex_texts: List[str]) -> np.ndarray:
    """
    Cross-encoder scores for (ans, exemplar_text) pairs with per-pair disk caching.
    """
    scores = np.empty(len(ex_texts), dtype=np.float32)
    to_score = []
    idxs = []
    for i, t in enumerate(ex_texts):
        p = _rerank_cache_path(qid, ans, t)
        if p.exists():
            try:
                scores[i] = float(p.read_text().strip())
                continue
            except Exception:
                pass
        to_score.append((ans, t))
        idxs.append(i)

    if to_score:
        ce = get_crossencoder()
        preds = ce.predict(to_score, batch_size=64, show_progress_bar=False)
        preds = np.array(preds, dtype=np.float32)
        for i, v in zip(idxs, preds.tolist()):
            scores[i] = float(v)
            try:
                _rerank_cache_path(qid, ans, ex_texts[i]).write_text(str(float(v)))
            except Exception:
                pass

    return scores


# =============================================================================
# MEANING-CLUSTER VOTE SCORER
# =============================================================================
def score_answer(qid: str, ans: str, ans_vec: np.ndarray, pack: ExemplarPack) -> ScoreDist:
    # retrieve
    sims = pack.vecs @ ans_vec  # (n,)
    n = sims.size
    topn = min(TOPN_RETRIEVE, n)
    idx = np.argpartition(-sims, topn - 1)[:topn]
    idx = idx[np.argsort(-sims[idx])]

    cand_texts = [pack.texts[i] for i in idx]
    cand_scores = pack.scores[idx].astype(np.int32)
    cand_vecs = pack.vecs[idx]

    # rerank topm
    topm = min(TOPM_RERANK, len(cand_texts))
    cand_texts_m = cand_texts[:topm]
    cand_scores_m = cand_scores[:topm]
    cand_vecs_m = cand_vecs[:topm]

    ce_scores = rerank_pairs_cached(qid, ans, cand_texts_m)  # (topm,)
    w = _softmax_temp(ce_scores, RERANK_TEMP)

    order = np.argsort(-ce_scores)
    ce_scores = ce_scores[order]
    w = w[order]
    cand_scores_m = cand_scores_m[order]
    cand_vecs_m = cand_vecs_m[order]
    cand_texts_m = [cand_texts_m[i] for i in order]

    # best coherent meaning cluster (seed-based threshold)
    seeds = list(range(min(SEED_TRIES, topm)))
    best_members = None
    best_mass = -1.0
    for s in seeds:
        seed_vec = cand_vecs_m[s]
        sim_to_seed = cand_vecs_m @ seed_vec
        members = np.where(sim_to_seed >= CLUSTER_SIM)[0]
        if members.size == 0:
            continue
        mass = float(np.sum(w[members]))
        if mass > best_mass:
            best_mass = mass
            best_members = members

    if best_members is None:
        best_members = np.arange(topm, dtype=np.int64)

    if best_members.size < MIN_CLUSTER:
        best_members = np.arange(min(topm, max(MIN_CLUSTER, 10)), dtype=np.int64)

    # vote inside cluster
    dist = np.zeros(4, dtype=np.float32)
    for i in best_members.tolist():
        s = int(cand_scores_m[i])
        if 0 <= s <= 3:
            dist[s] += float(w[i])

    if float(dist.sum()) <= 1e-8:
        top_label = int(cand_scores_m[0]) if topm else 1
        top_label = int(np.clip(top_label, 0, 3))
        dist[top_label] = 1.0

    dist = dist / (dist.sum() + 1e-9)
    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())
    max_sim = float(sims[idx[0]]) if len(idx) else 0.0
    return ScoreDist(dist=dist, expected=expected, pred=pred, conf=conf, max_sim=max_sim, method="retrieve+rerank+cluster_vote")


# =============================================================================
# KOBO + MAPPING
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
        except Exception as e:
            st.error(f"Kobo fetch failed: {type(e).__name__}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_mapping_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "prompt_hint" not in df.columns and "column" in df.columns:
        df = df.rename(columns={"column": "prompt_hint"})
    required = {"question_id", "attribute"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"mapping.csv must include: {', '.join(sorted(required))}")
    return df

_QID_PREFIX_TO_SECTION = {"LAV":"A1","II":"A2","EP":"A3","CFC":"A4","FTD":"A5","LDA":"A6","RDM":"A7"}

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
            prefix = qid.split("_Q", 1)[0]
            qn = int(qid.split("_Q")[-1])
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
# OFF-TOPIC (VERY CONSERVATIVE)
# =============================================================================
def conservative_offtopic(ans_vec: np.ndarray, qref: str) -> bool:
    qref = clean(qref)
    if not qref:
        return False
    q_emb = embed_texts([qref], kind="q").get(qref)
    if q_emb is None:
        return False
    qsim = float(np.dot(ans_vec, q_emb))
    return qsim < OFFTOPIC_QSIM


# =============================================================================
# DATAFRAME SCORING
# =============================================================================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame, packs: Dict[str, ExemplarPack]) -> pd.DataFrame:
    df_cols = list(df.columns)

    def want_col(c):
        lc = c.strip().lower()
        return any(h in lc for h in PASSTHROUGH_HINTS)

    passthrough_cols = [c for c in df_cols if want_col(c)]
    seen = set()
    passthrough_cols = [x for x in passthrough_cols if not (x in seen or seen.add(x))]

    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id", "staff_id", "staffid")), None)
    care_staff_col = next((c for c in df.columns if c.strip().lower() in ("care_staff", "care staff", "care-staff")), None)

    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start"] if c in df.columns), None)
    end_col = next((c for c in ["end"] if c in df.columns), None)

    n_rows = len(df)

    dt_series = pd.to_datetime(df[date_col].astype(str).str.strip().str.lstrip(","), errors="coerce") if date_col in df.columns else pd.Series([pd.NaT]*n_rows)

    if start_col:
        start_dt = pd.to_datetime(df[start_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT]*n_rows)
    if end_col:
        end_dt = pd.to_datetime(df[end_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT]*n_rows)
    duration_min = ((end_dt - start_dt).dt.total_seconds() / 60.0).clip(lower=0)

    rows = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute","")) in ORDERED_ATTRS]

    resolved_for_qid: Dict[str, str] = {}
    for r in rows:
        qid = clean(r.get("question_id",""))
        qhint = r.get("prompt_hint","") or r.get("column","")
        col = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if col:
            resolved_for_qid[qid] = col

    # embed unique answers once
    answers = []
    for _, rec in df.iterrows():
        for r in rows:
            qid = clean(r.get("question_id",""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(rec.get(col,""))
                if a:
                    answers.append(a)
    emb_map = embed_texts(list(set(answers)), kind="q")

    out_rows = []
    for i, rec in df.iterrows():
        row = {}
        row["Date"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_series.iloc[i]) else str(i)
        row["Duration"] = int(round(duration_min.iloc[i])) if not pd.isna(duration_min.iloc[i]) else ""

        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(rec.get(who_col)) if who_col else ""

        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date","Duration","Care_Staff"):
                continue
            row[c] = rec.get(c,"")

        per_attr: Dict[str, List[int]] = {}
        needs_review = False

        for r in rows:
            qid = clean(r.get("question_id",""))
            attr = clean(r.get("attribute",""))
            col = resolved_for_qid.get(qid)
            if not col or col not in df.columns:
                continue
            ans = clean(rec.get(col,""))
            if not ans:
                continue
            ans_vec = emb_map.get(ans)
            if ans_vec is None:
                continue

            pack = packs.get(qid)
            if pack is None or pack.vecs.size == 0:
                sc = 1
                row[f"{attr}_Qn{int(qid.split('_Q')[-1])}"] = sc if "_Q" in qid else sc
                continue

            sd = score_answer(qid, ans, ans_vec, pack)

            # off-topic cap (rare)
            qref = pack.qtext or (r.get("prompt_hint","") or "")
            if conservative_offtopic(ans_vec, qref):
                needs_review = True
                if sd.pred > OFFTOPIC_CAP:
                    sd = ScoreDist(dist=sd.dist, expected=sd.expected, pred=int(OFFTOPIC_CAP), conf=sd.conf, max_sim=sd.max_sim, method=sd.method+"+offtopic_cap")

            qn = None
            if "_Q" in qid:
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None

            if qn in (1,2,3,4):
                row[f"{attr}_Qn{qn}"] = int(sd.pred)
                row[f"{attr}_Rubric_Qn{qn}"] = BANDS[int(sd.pred)]
                per_attr.setdefault(attr, []).append(int(sd.pred))

        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")

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
        row["Needs_Review"] = bool(needs_review)
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    ordered = [c for c in ["Date","Duration","Care_Staff"] if c in res.columns]
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date","Duration","Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    mid_q = []
    for attr in ORDERED_ATTRS:
        for qn in (1,2,3,4):
            mid_q += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
    ordered += [c for c in mid_q if c in res.columns]

    mid_a = []
    for attr in ORDERED_ATTRS:
        mid_a += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += [c for c in mid_a if c in res.columns]

    ordered += [c for c in ["Overall Total (0–21)", "Needs_Review"] if c in res.columns]
    return res.reindex(columns=[c for c in ordered if c in res.columns])


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
            body={"valueInputOption": "USER_ENTERED", "data": [{"range": f"'{ws.title}'!A1", "values": [header] + values}]}
        )
        return True, f"✅ Wrote {len(values)} rows × {len(header)} cols to '{ws.title}'."
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()

    st.markdown("""
        <div class="app-header-card">
            <div class="pill">Thought Leadership • Auto Scoring</div>
            <h1>Thought Leadership</h1>
            <p class="app-header-subtitle">
                Importing Kobo submissions, scoring CARE thought leadership attributes (nearest exemplars),
                flagging AI-like responses, and exporting results to Google Sheets.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 1) Fetch Kobo FIRST so users see data immediately.
    with st.spinner("Fetching Kobo submissions…"):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found (or Kobo credentials missing).")
        return

    st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns):,}")
    st.dataframe(df, use_container_width=True)

    # 2) Load mapping + exemplars (local files) next.
    try:
        mapping = load_mapping_from_path(MAPPING_PATH)
    except Exception as e:
        st.error(f"Failed to load mapping: {e}")
        return

    try:
        exemplars = read_jsonl_path(EXEMPLARS_PATH)
        if not exemplars:
            st.error("Exemplars JSONL is empty.")
            return
    except Exception as e:
        st.error(f"Failed to read exemplars: {e}")
        return

    # 3) Build packs (cached). This can be slow on the first run, fast afterwards.
    with st.spinner("Preparing exemplar index (cached)…"):
        packs = build_packs(exemplars)

    # 4) Score
    with st.spinner("Scoring (retrieve → rerank → meaning-cluster vote)…"):
        scored = score_dataframe(df, mapping, packs)

    st.success("✅ Done.")
    st.dataframe(scored, use_container_width=True)

    st.download_button(
        "Download Excel",
        data=to_excel_bytes(scored),
        file_name="ThoughtLeadership_Scored.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.download_button(
        "Download CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="ThoughtLeadership_Scored.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if AUTO_PUSH:
        ok, msg = upload_df_to_gsheets(scored)
        (st.success if ok else st.error)(msg)


if __name__ == "__main__":
    main()
