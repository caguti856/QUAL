# thoughtleadership.py — Kobo mapping + Meaning scoring (Option B: Top-K weighted vote)
# Scores ANSWER TEXT against exemplar TEXT per question_id (no centroid, no review flags)

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

from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

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
        h1 { font-size: 2.0rem; font-weight: 750; }
        .app-header-card {
            position: relative;
            background: radial-gradient(circle at top left, rgba(242,106,33,0.15), rgba(250,204,21,0.06), #ffffff);
            border-radius: 1.25rem;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.12);
            margin-bottom: 1.1rem;
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

ORDERED_ATTRS = [
    "Locally Anchored Visioning",
    "Innovation and Insight",
    "Execution Planning",
    "Cross-Functional Collaboration",
    "Follow-Through Discipline",
    "Learning-Driven Adjustment",
    "Result-Oriented Decision-Making",
]

OVERALL_BANDS = [
    ("Exemplary Thought Leader", 19, 21),
    ("Strategic Advisor", 14, 18),
    ("Emerging Advisor", 8, 13),
    ("Needs Capacity Support", 0, 7),
]

# Kobo column resolution helpers
_QID_PREFIX_TO_SECTION = {"LAV": "A1", "II": "A2", "EP": "A3", "CFC": "A4", "FTD": "A5", "LDA": "A6", "RDM": "A7"}

PASSTHROUGH_HINTS = [
    "staff id", "staff_id", "staffid", "_id", "id", "_uuid", "uuid", "instanceid", "_submission_time",
    "submissiondate", "submission_date", "start", "_start", "end", "_end", "today", "date", "deviceid",
    "username", "enumerator", "submitted_via_web", "_xform_id_string", "formid", "assetid"
]
_EXCLUDE_SOURCE_COLS_LOWER = {
    "_id", "formhub/uuid", "start", "end", "today", "staff_id", "meta/instanceid",
    "_xform_id_string", "_uuid", "meta/rootuuid", "_submission_time", "_validation_status"
}

WORD_RX = re.compile(r"\w+")


# =============================================================================
# CLEANING
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
# KOBO
# =============================================================================
def kobo_url(asset_uid: str, kind: str = "submissions") -> str:
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
# MAPPING
# =============================================================================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # normalize expected columns
    if "prompt_hint" not in df.columns and "column" in df.columns:
        df = df.rename(columns={"column": "prompt_hint"})

    required = {"question_id", "attribute"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"mapping.csv must include: {', '.join(sorted(required))}")

    # snap attribute names to ORDERED_ATTRS
    norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
    target = {norm(a): a for a in ORDERED_ATTRS}

    def snap_attr(a):
        key = norm(a)
        if key in target:
            return target[key]
        best = process.extractOne(key, list(target.keys()), scorer=fuzz.token_set_ratio)
        return target[best[0]] if best and best[1] >= 75 else None

    df["attribute"] = df["attribute"].apply(snap_attr)
    df = df[df["attribute"].notna()].copy()
    if "prompt_hint" not in df.columns:
        df["prompt_hint"] = ""

    df["question_id"] = df["question_id"].astype(str).map(clean)
    df["prompt_hint"] = df["prompt_hint"].astype(str).map(clean)
    return df


def _score_kobo_header(col: str, token: str) -> int:
    col = clean(col).lower()
    token = clean(token).lower()
    if not col or not token:
        return 0
    if token in col:
        return 100
    return int(fuzz.token_set_ratio(token, col))


def resolve_kobo_column_for_mapping(df_cols: List[str], qid: str, prompt_hint: str) -> Optional[str]:
    """
    Maps question_id -> Kobo column in the fetched dataframe.
    Works with tokens like A1_1, A2_3, etc and fuzzy fallback to prompt_hint.
    """
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
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=6)
        for col, score, _ in hits:
            if score >= 80:
                return col

    return None


# =============================================================================
# EXEMPLARS
# =============================================================================
def read_jsonl_path(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exemplars file not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith(","):
                s = s.lstrip(",").strip()
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                # tolerate trailing comma
                s2 = re.sub(r",\s*$", "", s)
                out.append(json.loads(s2))
    return out


@dataclass
class ExemplarPack:
    vecs: np.ndarray       # (n, d) normalized
    scores: np.ndarray     # (n,) int 0..3
    texts: List[str]       # exemplar text


# =============================================================================
# EMBEDDINGS (fast + cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    # Small & fast semantic model
    return SentenceTransformer(st.secrets.get("EMBED_MODEL", "all-MiniLM-L6-v2"))


@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: Tuple[str, ...]) -> np.ndarray:
    model = get_embedder()
    embs = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)
    return embs


def embed_map(texts: List[str]) -> Dict[str, np.ndarray]:
    uniq = []
    seen = set()
    for t in texts:
        t = clean(t)
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return {}
    embs = embed_texts_cached(tuple(uniq))
    return {t: v for t, v in zip(uniq, embs)}


def build_packs_by_question(exemplars: List[dict]) -> Dict[str, ExemplarPack]:
    """
    Build pack per question_id using exemplar 'text' ONLY.
    (No centroids. Every answer matches to real exemplar texts.)
    """
    by_qid: Dict[str, Dict[str, list]] = {}
    all_texts: List[str] = []

    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        txt = clean(e.get("text", ""))
        if not qid or not txt:
            continue
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0
        sc = int(np.clip(sc, 0, 3))
        d = by_qid.setdefault(qid, {"texts": [], "scores": []})
        d["texts"].append(txt)
        d["scores"].append(sc)
        all_texts.append(txt)

    emb = embed_map(list(set(all_texts)))

    packs: Dict[str, ExemplarPack] = {}
    for qid, d in by_qid.items():
        # de-dup exact text+score pairs
        seen = set()
        texts, scores, vecs = [], [], []
        for t, s in zip(d["texts"], d["scores"]):
            key = (t, int(s))
            if key in seen:
                continue
            seen.add(key)
            v = emb.get(t)
            if v is None:
                continue
            texts.append(t)
            scores.append(int(s))
            vecs.append(v)

        if not vecs:
            packs[qid] = ExemplarPack(vecs=np.zeros((0, 384), dtype=np.float32), scores=np.array([], dtype=np.int32), texts=[])
        else:
            packs[qid] = ExemplarPack(
                vecs=np.vstack(vecs).astype(np.float32),
                scores=np.array(scores, dtype=np.int32),
                texts=texts,
            )

    return packs


# =============================================================================
# OPTION B: Top-K weighted vote scoring (meaning-based)
# =============================================================================
def softmax(x: np.ndarray, temp: float) -> np.ndarray:
    t = float(max(1e-6, temp))
    z = (x - x.max()) / t
    ex = np.exp(z).astype(np.float32)
    return ex / (ex.sum() + 1e-9)


def score_against_pack_vote(
    pack: ExemplarPack,
    ans_vec: np.ndarray,
    k: int = 7,
    temp: float = 0.08,
) -> Tuple[Optional[int], float, str, float]:
    """
    Returns:
      pred_score (0..3 or None),
      conf (max class weight),
      best_match_text,
      best_match_sim
    """
    if pack is None or pack.vecs.size == 0 or ans_vec is None:
        return None, 0.0, "", 0.0

    sims = pack.vecs @ ans_vec  # cosine similarity, vecs normalized
    if sims.size == 0:
        return None, 0.0, "", 0.0

    # best match (for traceability)
    best_i = int(np.argmax(sims))
    best_text = pack.texts[best_i]
    best_sim = float(sims[best_i])

    # vote over top-k
    kk = int(max(1, min(k, sims.size)))
    idx = np.argpartition(-sims, kk - 1)[:kk]
    idx = idx[np.argsort(-sims[idx])]

    top_sims = sims[idx]
    top_scores = pack.scores[idx]

    w = softmax(top_sims.astype(np.float32), temp=temp)

    class_w = np.zeros(4, dtype=np.float32)
    for sc, wi in zip(top_scores.tolist(), w.tolist()):
        if 0 <= int(sc) <= 3:
            class_w[int(sc)] += float(wi)

    pred = int(class_w.argmax())
    conf = float(class_w.max())
    return pred, conf, best_text, best_sim


# =============================================================================
# DATAFRAME SCORING
# =============================================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return bio.getvalue()


def score_dataframe(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    packs_by_qid: Dict[str, ExemplarPack],
    knn_k: int,
    knn_temp: float,
    include_match_cols: bool = True,
) -> pd.DataFrame:
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

    start_col = next((c for c in ["start"] if c in df.columns), None)
    end_col = next((c for c in ["end"] if c in df.columns), None)

    n_rows = len(df)

    # Parse date
    dt_series = pd.to_datetime(df[date_col].astype(str).str.strip().str.lstrip(","), errors="coerce") if date_col in df.columns else pd.Series([pd.NaT] * n_rows)

    # Duration
    if start_col:
        start_dt = pd.to_datetime(df[start_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)

    if end_col:
        end_dt = pd.to_datetime(df[end_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT] * n_rows)

    duration_min = ((end_dt - start_dt).dt.total_seconds() / 60.0).clip(lower=0)

    # Resolve mapping -> Kobo columns
    rows = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute", "")) in ORDERED_ATTRS]

    resolved_for_qid: Dict[str, str] = {}
    missing_qids = []
    for r in rows:
        qid = clean(r.get("question_id", ""))
        hint = r.get("prompt_hint", "") or ""
        col = resolve_kobo_column_for_mapping(df_cols, qid, hint)
        if col:
            resolved_for_qid[qid] = col
        else:
            missing_qids.append(qid)

    # Batch-embed ALL unique answers across df for speed
    all_answers = []
    for _, rec in df.iterrows():
        for r in rows:
            qid = clean(r.get("question_id", ""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(rec.get(col, ""))
                if a:
                    all_answers.append(a)

    ans_emb = embed_map(list(set(all_answers)))

    # cache exact same (qid, answer) scoring
    exact_cache: Dict[Tuple[str, str], Tuple[int, float, str, float]] = {}

    out_rows = []
    for i, rec in df.iterrows():
        row = {}
        row["Date"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_series.iloc[i]) else str(i)
        row["Duration"] = int(round(duration_min.iloc[i])) if not pd.isna(duration_min.iloc[i]) else ""

        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(rec.get(who_col)) if who_col else ""

        # passthrough
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date", "Duration", "Care_Staff"):
                continue
            row[c] = rec.get(c, "")

        per_attr: Dict[str, List[int]] = {}

        # per-question scoring
        for r in rows:
            qid = clean(r.get("question_id", ""))
            attr = clean(r.get("attribute", ""))
            col = resolved_for_qid.get(qid)
            if not col or col not in df.columns:
                continue

            ans = clean(rec.get(col, ""))
            if not ans:
                continue

            qn = None
            if "_Q" in qid:
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None
            if qn not in (1, 2, 3, 4):
                continue

            cache_key = (qid, ans)
            if cache_key in exact_cache:
                sc, conf, best_txt, best_sim = exact_cache[cache_key]
            else:
                vec = ans_emb.get(ans)
                pack = packs_by_qid.get(qid)
                sc, conf, best_txt, best_sim = score_against_pack_vote(
                    pack=pack,
                    ans_vec=vec,
                    k=knn_k,
                    temp=knn_temp,
                )
                if sc is None:
                    sc = 1  # safe default if no pack/embedding
                exact_cache[cache_key] = (int(sc), float(conf), str(best_txt), float(best_sim))

            row[f"{attr}_Qn{qn}"] = int(sc)
            row[f"{attr}_Rubric_Qn{qn}"] = BANDS[int(sc)]
            if include_match_cols:
                row[f"{attr}_Qn{qn}_MatchText"] = best_txt
                row[f"{attr}_Qn{qn}_MatchSim"] = round(float(best_sim), 4)

            per_attr.setdefault(attr, []).append(int(sc))

        # fill blanks (so table is consistent)
        for attr in ORDERED_ATTRS:
            for qn in (1, 2, 3, 4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")
                if include_match_cols:
                    row.setdefault(f"{attr}_Qn{qn}_MatchText", "")
                    row.setdefault(f"{attr}_Qn{qn}_MatchSim", "")

        # attribute averages + total
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
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # column ordering (keep it readable)
    ordered = [c for c in ["Date", "Duration", "Care_Staff"] if c in res.columns]
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date", "Duration", "Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    q_cols = []
    for attr in ORDERED_ATTRS:
        for qn in (1, 2, 3, 4):
            q_cols += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
            if include_match_cols:
                q_cols += [f"{attr}_Qn{qn}_MatchText", f"{attr}_Qn{qn}_MatchSim"]
    ordered += [c for c in q_cols if c in res.columns]

    a_cols = []
    for attr in ORDERED_ATTRS:
        a_cols += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += [c for c in a_cols if c in res.columns]

    ordered += [c for c in ["Overall Total (0–21)", "Overall Rank"] if c in res.columns]
    res = res.reindex(columns=[c for c in ordered if c in res.columns])

    if missing_qids:
        # show once in UI via session state
        st.session_state["missing_qids"] = sorted(set(missing_qids))

    return res


# =============================================================================
# GOOGLE SHEETS (optional)
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
        return sh.add_worksheet(title=DEFAULT_WS_NAME, rows="20000", cols="200")


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

    st.markdown(
        """
        <div class="app-header-card">
            <div class="pill">Thought Leadership • Meaning Scoring (Option B)</div>
            <h1>Thought Leadership</h1>
            <p style="color:#6b7280;margin:0;">
                Scores each response by comparing the <b>answer text</b> to exemplar <b>text</b> per <b>question_id</b>,
                using a <b>Top-K weighted vote</b>. No centroids. No review flags.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Scoring controls")
        knn_k = st.slider("Top-K exemplars", min_value=3, max_value=25, value=int(st.secrets.get("KNN_K", 7)), step=1)
        knn_temp = st.slider("Softmax temperature", min_value=0.02, max_value=0.30, value=float(st.secrets.get("KNN_TEMP", 0.08)), step=0.01)
        include_match_cols = st.checkbox("Include matched exemplar text + similarity columns", value=True)

    # Load mapping + exemplars
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

    with st.spinner("Building per-question exemplar packs (cached)…"):
        packs_by_qid = build_packs_by_question(exemplars)

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

    with st.spinner("Scoring (Top-K vote per question)…"):
        scored = score_dataframe(
            df=df,
            mapping=mapping,
            packs_by_qid=packs_by_qid,
            knn_k=knn_k,
            knn_temp=knn_temp,
            include_match_cols=include_match_cols,
        )

    # Show mapping gaps (informational only)
    missing_qids = st.session_state.get("missing_qids", [])
    if missing_qids:
        st.info(f"Some mapping question_id(s) could not be matched to Kobo columns: {', '.join(missing_qids[:12])}"
                + (" …" if len(missing_qids) > 12 else ""))

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
