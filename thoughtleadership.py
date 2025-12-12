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

from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, CrossEncoder


# =============================================================================
# UI / STYLING (minimal but nice)
# =============================================================================
def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --primary: #F26A21; /* CARE orange */
            --bg: #F7F7F8;
            --card: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --border: #e5e7eb;
        }
        [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
        .section-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 14px;
            margin: 10px 0 14px 0;
        }
        .pill {
            display: inline-block; font-size: 0.75rem; padding: 0.15rem 0.7rem; border-radius: 999px;
            background: rgba(242,106,33,0.08); border: 1px solid rgba(242,106,33,0.55);
            color: #9A3412;
        }
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

DATASETS_DIR = Path("DATASETS")
MAPPING_PATH = DATASETS_DIR / "mapping1.csv"
EXEMPLARS_PATH = DATASETS_DIR / "thought_leadership.cleaned.jsonl"



# =============================================================================
# CONFIG (fast defaults for your dataset size)
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

# Retrieve + rerank sizes (fast + accurate)
TOPN_RETRIEVE = int(st.secrets.get("TOPN_RETRIEVE", 50))   # embedding shortlist
TOPM_RERANK = int(st.secrets.get("TOPM_RERANK", 12))       # cross-encoder rerank inside shortlist
RERANK_BATCH = int(st.secrets.get("RERANK_BATCH", 96))     # cross-encoder batch size

# Meaning cluster vote
SEED_TRIES = int(st.secrets.get("SEED_TRIES", 4))
CLUSTER_SIM = float(st.secrets.get("CLUSTER_SIM", 0.78))
MIN_CLUSTER = int(st.secrets.get("MIN_CLUSTER", 3))
RERANK_TEMP = float(st.secrets.get("RERANK_TEMP", 0.35))

# Off-topic: flag only (do not change score)
OFFTOPIC_FLAG_QSIM = float(st.secrets.get("OFFTOPIC_FLAG_QSIM", 0.05))

# Model choices (allow overriding in secrets)
# If speed is still an issue, change BI_ENCODER_NAME to "intfloat/e5-small-v2"
BI_ENCODER_NAME = st.secrets.get("BI_ENCODER_NAME", "intfloat/e5-base-v2")
CROSS_ENCODER_NAME = st.secrets.get("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Cache directories (single files, not many tiny files)
CACHE_DIR = Path(st.secrets.get("TL_CACHE_DIR", ".tl_cache"))
PACK_DIR = CACHE_DIR / "packs"
EMB_DIR = CACHE_DIR / "emb"
for d in (PACK_DIR, EMB_DIR):
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


# =============================================================================
# HELPERS
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
    z = (x / t) - np.max(x / t)
    ex = np.exp(z)
    return ex / (np.sum(ex) + 1e-9)


# =============================================================================
# MODELS (cached in memory)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_biencoder() -> SentenceTransformer:
    return SentenceTransformer(BI_ENCODER_NAME)

@st.cache_resource(show_spinner=False)
def get_crossencoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_NAME)

def _is_e5_model() -> bool:
    return "e5" in BI_ENCODER_NAME.lower()

def _e5_query(t: str) -> str:
    return f"query: {t}"

def _e5_passage(t: str) -> str:
    return f"passage: {t}"


# =============================================================================
# EMBEDDINGS (disk cached, but as .npy per text — manageable at your size)
# =============================================================================
def _emb_cache_path(text: str, kind: str) -> Path:
    # kind: "q" (answers/questions), "p" (exemplar passages)
    key = _sha1(kind + "|" + BI_ENCODER_NAME + "|" + clean(text))
    return EMB_DIR / f"{key}.npy"

def embed_texts(texts: List[str], kind: str) -> Dict[str, np.ndarray]:
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
            enc_in = [_e5_query(t) if kind == "q" else _e5_passage(t) for t in missing]
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
# EXEMPLAR PACKS (per question_id; embeds ONLY exemplar 'text')
# =============================================================================
@dataclass
class ExemplarPack:
    vecs: np.ndarray           # (n, d) normalized
    scores: np.ndarray         # (n,) int 0..3
    texts: List[str]           # exemplar answers ("text")
    qtext: str                 # question_text (for flagging only)
    attr: str                  # attribute label

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

def _pack_cache_path(qid: str) -> Path:
    return PACK_DIR / f"{_sha1(BI_ENCODER_NAME + '|pack|' + qid)}.npz"

def build_packs(exemplars: List[dict]) -> Dict[str, ExemplarPack]:
    by_q: Dict[str, dict] = {}
    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        if not qid:
            continue
        txt = clean(e.get("text", ""))  # IMPORTANT: compare against "text" ONLY
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
        cachep = _pack_cache_path(qid)
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
        except Exception as e:
            st.error(f"Kobo fetch failed: {type(e).__name__}: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


# =============================================================================
# MAPPING + KOBO COLUMN RESOLUTION
# =============================================================================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
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
# SCORING CORE (batched per question_id)
# =============================================================================
@dataclass
class ScoreResult:
    pred: int
    conf: float
    needs_review: bool

def _best_cluster_vote(
    cand_vecs: np.ndarray,      # (M, d)
    cand_scores: np.ndarray,    # (M,)
    ce_scores: np.ndarray,      # (M,)
) -> Tuple[int, float]:
    # Sort by reranker
    order = np.argsort(-ce_scores)
    ce_scores = ce_scores[order]
    cand_scores = cand_scores[order]
    cand_vecs = cand_vecs[order]

    w = _softmax_temp(ce_scores, RERANK_TEMP)  # weights per candidate

    topm = cand_vecs.shape[0]
    seeds = range(min(SEED_TRIES, topm))
    best_members = None
    best_mass = -1.0

    for s in seeds:
        seed = cand_vecs[s]
        sim = cand_vecs @ seed
        members = np.where(sim >= CLUSTER_SIM)[0]
        if members.size == 0:
            continue
        mass = float(np.sum(w[members]))
        if mass > best_mass:
            best_mass = mass
            best_members = members

    if best_members is None:
        best_members = np.arange(topm, dtype=np.int64)

    if best_members.size < MIN_CLUSTER:
        # fallback: trust top candidates rather than expanding to all
        best_members = np.arange(min(topm, max(MIN_CLUSTER, 8)), dtype=np.int64)

    dist = np.zeros(4, dtype=np.float32)
    for i in best_members.tolist():
        s = int(cand_scores[i])
        if 0 <= s <= 3:
            dist[s] += float(w[i])

    if float(dist.sum()) <= 1e-8:
        # fallback to top label
        top_label = int(np.clip(int(cand_scores[0]), 0, 3))
        dist[top_label] = 1.0

    dist = dist / (dist.sum() + 1e-9)
    expected = float(np.dot(dist, np.arange(4)))
    pred = int(np.clip(int(round(expected)), 0, 3))
    conf = float(dist.max())
    return pred, conf

def _score_question_batched(
    qid: str,
    answers: List[str],              # len = N (one per submission row; may include "")
    answer_vecs: List[Optional[np.ndarray]],
    pack: ExemplarPack,
) -> List[Optional[ScoreResult]]:
    """
    Scores all answers for one qid in a batched way:
      - retrieve topN by embeddings
      - rerank topM by cross-encoder in one big batch
      - per-answer coherent cluster vote
    Returns list aligned with answers length (None where answer missing).
    """
    ce = get_crossencoder()

    # Collect valid indices
    valid_idx = [i for i, (a, v) in enumerate(zip(answers, answer_vecs)) if a and v is not None]
    if not valid_idx:
        return [None] * len(answers)

    A = np.vstack([answer_vecs[i] for i in valid_idx]).astype(np.float32)  # (Nv, d)

    # retrieve
    sims = pack.vecs @ A.T  # (n_ex, Nv)
    n_ex, Nv = sims.shape
    topn = min(TOPN_RETRIEVE, n_ex)
    # indices of topn per column
    idx_topn = np.argpartition(-sims, topn - 1, axis=0)[:topn, :]  # (topn, Nv)

    # Now build rerank pairs for topM per answer
    pairs = []
    meta = []  # (local_answer_j, candidate_k, exemplar_index)
    for j in range(Nv):
        inds = idx_topn[:, j]
        # sort those by similarity
        inds = inds[np.argsort(-sims[inds, j])]
        inds = inds[: min(TOPM_RERANK, len(inds))]
        ans_text = answers[valid_idx[j]]
        for k, ex_i in enumerate(inds.tolist()):
            pairs.append((ans_text, pack.texts[ex_i]))
            meta.append((j, k, ex_i))

    if not pairs:
        return [None] * len(answers)

    # batched rerank
    ce_scores_flat = np.array(
        ce.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False),
        dtype=np.float32,
    )

    # group scores back per answer
    # Build structure: for each j, keep list of (ex_i, ce_score)
    per_j: List[List[Tuple[int, float]]] = [[] for _ in range(Nv)]
    for (j, k, ex_i), sc in zip(meta, ce_scores_flat.tolist()):
        per_j[j].append((ex_i, float(sc)))

    # precompute question embedding for review flagging (meaning only; no score change)
    qref = clean(pack.qtext)
    qref_vec = embed_texts([qref], kind="q").get(qref) if qref else None

    out: List[Optional[ScoreResult]] = [None] * len(answers)

    for j in range(Nv):
        items = per_j[j]
        if not items:
            continue

        # build candidate arrays
        ex_inds = np.array([x[0] for x in items], dtype=np.int32)
        ce_s = np.array([x[1] for x in items], dtype=np.float32)

        cand_vecs = pack.vecs[ex_inds]
        cand_scores = pack.scores[ex_inds]

        pred, conf = _best_cluster_vote(cand_vecs, cand_scores, ce_s)

        # flag off-topic ONLY (does not change score)
        needs_review = False
        if qref_vec is not None:
            av = answer_vecs[valid_idx[j]]
            if av is not None:
                qsim = float(np.dot(av, qref_vec))
                if qsim < OFFTOPIC_FLAG_QSIM:
                    needs_review = True

        out[valid_idx[j]] = ScoreResult(pred=int(pred), conf=float(conf), needs_review=needs_review)

    return out


# =============================================================================
# DATAFRAME SCORING
# =============================================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return bio.getvalue()

def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame, packs: Dict[str, ExemplarPack]) -> pd.DataFrame:
    df_cols = list(df.columns)

    def want_col(c: str) -> bool:
        lc = c.strip().lower()
        return any(h in lc for h in PASSTHROUGH_HINTS)

    passthrough_cols = [c for c in df_cols if want_col(c)]
    seen = set()
    passthrough_cols = [x for x in passthrough_cols if not (x in seen or seen.add(x))]

    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id", "staff_id", "staffid")), None)
    care_staff_col = next((c for c in df.columns if c.strip().lower() in ("care_staff", "care staff", "care-staff")), None)

    date_cols_pref = ["_submission_time", "SubmissionDate", "submissiondate", "end", "End", "start", "Start", "today", "date", "Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = "start" if "start" in df.columns else None
    end_col = "end" if "end" in df.columns else None

    n_rows = len(df)
    dt_series = pd.to_datetime(df[date_col].astype(str).str.strip().str.lstrip(","), errors="coerce") if date_col in df.columns else pd.Series([pd.NaT] * n_rows)

    if start_col:
        start_dt = pd.to_datetime(df[start_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)
    if end_col:
        end_dt = pd.to_datetime(df[end_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT] * n_rows)
    duration_min = ((end_dt - start_dt).dt.total_seconds() / 60.0).clip(lower=0)

    rows = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute", "")) in ORDERED_ATTRS]

    # Resolve Kobo response columns for each qid
    resolved_for_qid: Dict[str, str] = {}
    qid_to_attr: Dict[str, str] = {}
    qid_to_hint: Dict[str, str] = {}
    for r in rows:
        qid = clean(r.get("question_id", ""))
        attr = clean(r.get("attribute", ""))
        hint = clean(r.get("prompt_hint", "") or r.get("column", ""))
        if not qid or not attr:
            continue
        col = resolve_kobo_column_for_mapping(df_cols, qid, hint)
        if col:
            resolved_for_qid[qid] = col
            qid_to_attr[qid] = attr
            qid_to_hint[qid] = hint

    # Build all answers per qid, and embed unique answer texts once
    answers_by_qid: Dict[str, List[str]] = {qid: [""] * n_rows for qid in resolved_for_qid.keys()}
    all_answer_texts: List[str] = []

    for qid, col in resolved_for_qid.items():
        for i in range(n_rows):
            a = clean(df.iloc[i].get(col, ""))
            answers_by_qid[qid][i] = a
            if a:
                all_answer_texts.append(a)

    # embed unique answers once
    emb_map = embed_texts(list(set(all_answer_texts)), kind="q")

    # Prepare output rows (metadata first)
    out_rows: List[dict] = []
    for i in range(n_rows):
        rec = df.iloc[i]
        row = {
            "Date": pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_series.iloc[i]) else str(i),
            "Duration": int(round(duration_min.iloc[i])) if not pd.isna(duration_min.iloc[i]) else "",
            "Care_Staff": str(rec.get(care_staff_col or staff_id_col)) if (care_staff_col or staff_id_col) else "",
        }
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date", "Duration", "Care_Staff"):
                continue
            row[c] = rec.get(c, "")
        out_rows.append(row)

    # Score per question in batches (fast)
    any_needs_review = [False] * n_rows
    per_attr_scores: List[Dict[str, List[int]]] = [dict() for _ in range(n_rows)]

    for qid, answers in answers_by_qid.items():
        pack = packs.get(qid)
        if pack is None or pack.vecs.size == 0:
            # no exemplars: leave blanks (or default 1 if you prefer)
            continue

        # align answer vecs with rows
        answer_vecs: List[Optional[np.ndarray]] = [emb_map.get(a) if a else None for a in answers]

        results = _score_question_batched(qid, answers, answer_vecs, pack)

        attr = qid_to_attr.get(qid, "")
        # determine question number for output columns
        qn = None
        if "_Q" in qid:
            try:
                qn = int(qid.split("_Q")[-1])
            except Exception:
                qn = None

        if qn not in (1, 2, 3, 4):
            continue

        for i, sr in enumerate(results):
            if sr is None:
                continue
            sc = int(sr.pred)
            out_rows[i][f"{attr}_Qn{qn}"] = sc
            out_rows[i][f"{attr}_Rubric_Qn{qn}"] = BANDS[sc]
            per_attr_scores[i].setdefault(attr, []).append(sc)
            if sr.needs_review:
                any_needs_review[i] = True

    # Fill blanks + compute averages and overall
    for i in range(n_rows):
        row = out_rows[i]
        overall_total = 0
        for attr in ORDERED_ATTRS:
            # ensure q columns exist
            for qn in (1, 2, 3, 4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")

            scores = per_attr_scores[i].get(attr, [])
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
        row["Needs_Review"] = bool(any_needs_review[i])

    res = pd.DataFrame(out_rows)

    # Order columns nicely
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

    ordered += [c for c in ["Overall Total (0–21)", "Needs_Review"] if c in res.columns]
    return res.reindex(columns=[c for c in ordered if c in res.columns])


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()

    st.markdown(
        """
        <div class="section-card">
          <div class="pill">Thought Leadership • Fast Meaning Scoring</div>
          <h2 style="margin:8px 0 0 0;">Thought Leadership Scoring</h2>
          <div style="color:#6b7280; margin-top:6px;">
            Scoring = meaning match between <b>response</b> and exemplar <b>text</b> only (not the question),
            using retrieve → rerank → coherent cluster vote.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1) Kobo first
    with st.spinner("Fetching Kobo submissions…"):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found (or Kobo secrets missing).")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
            st.error("Exemplars JSONL is empty.")
            return
    except Exception as e:
        st.error(f"Failed to read exemplars: {e}")
        return

    # Button to start scoring (prevents reruns from auto-triggering long work)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("Models:", BI_ENCODER_NAME, " + ", CROSS_ENCODER_NAME)
    st.write(f"Retrieve TOPN={TOPN_RETRIEVE}, Rerank TOPM={TOPM_RERANK}, ClusterSim={CLUSTER_SIM}")
    go = st.button("Score now", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)
    if not go:
        return

    # 3) Build packs (cached)
    with st.spinner("Preparing exemplar packs (cached as single files)…"):
        packs = build_packs(exemplars)

    # 4) Score
    with st.spinner("Scoring (meaning-based)…"):
        scored = score_dataframe(df, mapping, packs)

    st.success("✅ Done.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.dataframe(scored, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download Excel",
        data=to_excel_bytes(scored),
        file_name="ThoughtLeadership_Scored.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    st.download_button(
        "Download CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="ThoughtLeadership_Scored.csv",
        mime="text/csv",
        width="stretch",
    )


if __name__ == "__main__":
    main()
