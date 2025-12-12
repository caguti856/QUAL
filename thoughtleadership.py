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
            --bg: #f6f7f9;
            --card: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --border: #e5e7eb;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #FFF7ED 0, #F9FAFB 40%, #F3F4F6 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #1f2937;
        }
        [data-testid="stSidebar"] * { color: #e5e7eb !important; }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .header-card {
            background: radial-gradient(circle at top left, rgba(242,106,33,0.15), rgba(250,204,21,0.06), #ffffff);
            border-radius: 1.25rem;
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.10);
            margin-bottom: 1.2rem;
        }
        .pill {
            display:inline-block;
            font-size: 0.75rem;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            background: rgba(242,106,33,0.08);
            border: 1px solid rgba(242,106,33,0.6);
            color: #9A3412;
            margin-bottom: 0.5rem;
        }

        .section-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.04);
            margin-bottom: 1rem;
        }

        .stDownloadButton button, .stButton button {
            border-radius: 999px !important;
            padding: 0.35rem 1.2rem !important;
            font-weight: 600 !important;
            border: 1px solid rgba(242,106,33,0.85) !important;
            background: linear-gradient(135deg, var(--primary) 0%, #FB923C 100%) !important;
            color: #FFFBEB !important;
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
# SCORING CONSTANTS (OPTION B — automatic)
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

# Option B scoring parameters (AUTO)
TOPK = 40                 # use up to 40 exemplars if available
TEMP = 1.0                # temperature for softmax vote
CLOSE_MARGIN = 0.02       # include extras within (cutoff - margin)
MAX_EXTRA = 40            # cap extra exemplars for speed

# Embedding model (small + fast)
EMBED_MODEL_NAME = st.secrets.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

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


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32)
    t = float(max(1e-6, temp))
    z = x / t
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / (np.sum(ez) + 1e-9)


# =============================================================================
# MODELS (cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


# =============================================================================
# EXEMPLARS (per question_id packs)
# =============================================================================
@dataclass
class ExemplarPack:
    vecs: np.ndarray      # (n, d) normalized
    scores: np.ndarray    # (n,) int in 0..3
    texts: List[str]      # exemplar texts
    attr: str             # attribute (for reference)


def read_jsonl_path(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith(","):
                s = s.lstrip(",").strip()
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                # tolerate trailing comma
                s2 = re.sub(r",\s*$", "", s)
                rows.append(json.loads(s2))
    return rows


def embed_text_list(texts: List[str]) -> Dict[str, np.ndarray]:
    """Batch embed unique texts, normalized vectors."""
    uniq = []
    seen = set()
    for t in texts:
        t = clean(t)
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    if not uniq:
        return {}

    model = get_embedder()
    vecs = model.encode(
        uniq,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)

    return {t: v for t, v in zip(uniq, vecs)}


def build_packs_by_qid(exemplars: List[dict]) -> Dict[str, ExemplarPack]:
    """
    Build packs strictly per question_id.
    Embeds ONLY exemplar['text'] (meaning comparison).
    """
    by_qid: Dict[str, Dict[str, object]] = {}
    all_texts: List[str] = []

    for e in exemplars:
        qid = clean(e.get("question_id", ""))
        txt = clean(e.get("text", ""))
        attr = clean(e.get("attribute", ""))

        if not qid or not txt:
            continue
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0
        sc = int(np.clip(sc, 0, 3))

        d = by_qid.setdefault(qid, {"texts": [], "scores": [], "attr": attr})
        d["texts"].append(txt)
        d["scores"].append(sc)
        if attr:
            d["attr"] = attr

        all_texts.append(txt)

    emb_map = embed_text_list(all_texts)

    packs: Dict[str, ExemplarPack] = {}
    for qid, d in by_qid.items():
        texts = [clean(t) for t in d["texts"] if clean(t)]
        scores = np.array([int(x) for x in d["scores"]], dtype=np.int32)

        # keep only texts that embedded successfully
        kept_texts = []
        kept_scores = []
        kept_vecs = []
        for t, s in zip(texts, scores.tolist()):
            v = emb_map.get(t)
            if v is None:
                continue
            kept_texts.append(t)
            kept_scores.append(int(s))
            kept_vecs.append(v)

        if kept_vecs:
            mat = np.vstack(kept_vecs).astype(np.float32)
        else:
            mat = np.zeros((0, 384), dtype=np.float32)

        packs[qid] = ExemplarPack(
            vecs=mat,
            scores=np.array(kept_scores, dtype=np.int32),
            texts=kept_texts,
            attr=str(d.get("attr", "")),
        )

    return packs


def score_against_pack_optionB(pack: ExemplarPack, ans_vec: np.ndarray) -> Tuple[Optional[int], float, str, float]:
    """
    Option B: Top-K softmax vote (temp=1.0) + include near-cutoff extras.
    Returns: (pred_score, conf, best_exemplar_text, best_sim)
    """
    if pack is None or pack.vecs.size == 0 or ans_vec is None:
        return None, 0.0, "", 0.0

    sims = pack.vecs @ ans_vec
    n = sims.size
    if n == 0:
        return None, 0.0, "", 0.0

    order = np.argsort(-sims)
    best_i = int(order[0])
    best_text = pack.texts[best_i]
    best_sim = float(sims[best_i])

    k = int(min(TOPK, n))
    base_idx = order[:k]

    # include near-cutoff extras when we have more than k exemplars
    idx = base_idx
    if n > k and k > 0:
        cutoff = float(sims[order[k - 1]])
        rest = order[k:]
        close_mask = sims[rest] >= (cutoff - CLOSE_MARGIN)
        extras = rest[close_mask][:MAX_EXTRA]
        if extras.size > 0:
            idx = np.concatenate([base_idx, extras])

    top_sims = sims[idx].astype(np.float32)
    top_scores = pack.scores[idx].astype(np.int32)

    w = softmax(top_sims, temp=TEMP)

    class_w = np.zeros(4, dtype=np.float32)
    for s, wi in zip(top_scores.tolist(), w.tolist()):
        if 0 <= int(s) <= 3:
            class_w[int(s)] += float(wi)

    pred = int(class_w.argmax())
    conf = float(class_w.max())
    return pred, conf, best_text, best_sim


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
    """
    Finds the Kobo column for this question_id using A1_1 patterns first, then fuzzy hint.
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
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=5)
        for col, score, _ in hits:
            if score >= 80:
                return col

    return None


# =============================================================================
# SCORING DATAFRAME
# =============================================================================
def score_dataframe(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    packs_by_qid: Dict[str, ExemplarPack],
    include_match_cols: bool = True,
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

    # keep only mapped attributes
    rows = [r for r in mapping.to_dict(orient="records") if clean(r.get("attribute", "")) in ORDERED_ATTRS]

    # resolve Kobo columns for each qid
    resolved_for_qid: Dict[str, str] = {}
    for r in rows:
        qid = clean(r.get("question_id", ""))
        qhint = r.get("prompt_hint", "") or r.get("column", "")
        col = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if col:
            resolved_for_qid[qid] = col

    # embed all unique answers used by mapped questions
    unique_answers: List[str] = []
    for _, rec in df.iterrows():
        for r in rows:
            qid = clean(r.get("question_id", ""))
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(rec.get(col, ""))
                if a:
                    unique_answers.append(a)

    unique_answers = list(set(unique_answers))
    ans_emb_map = embed_text_list(unique_answers)

    out_rows = []
    for i, rec in df.iterrows():
        row = {}
        row["Date"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_series.iloc[i]) else str(i)
        row["Duration"] = int(round(duration_min.iloc[i])) if not pd.isna(duration_min.iloc[i]) else ""
        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(rec.get(who_col)) if who_col else ""

        # passthrough useful fields
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date", "Duration", "Care_Staff"):
                continue
            row[c] = rec.get(c, "")

        per_attr: Dict[str, List[int]] = {}

        for r in rows:
            qid = clean(r.get("question_id", ""))
            attr = clean(r.get("attribute", ""))
            col = resolved_for_qid.get(qid)
            if not col or col not in df.columns:
                continue

            ans = clean(rec.get(col, ""))
            if not ans:
                continue

            ans_vec = ans_emb_map.get(ans)
            if ans_vec is None:
                continue

            pack = packs_by_qid.get(qid)
            if pack is None:
                continue

            pred, conf, best_text, best_sim = score_against_pack_optionB(pack, ans_vec)
            if pred is None:
                continue

            qn = None
            if "_Q" in qid:
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None

            if qn in (1, 2, 3, 4):
                row[f"{attr}_Qn{qn}"] = int(pred)
                row[f"{attr}_Rubric_Qn{qn}"] = BANDS[int(pred)]
                per_attr.setdefault(attr, []).append(int(pred))

                if include_match_cols:
                    row[f"{attr}_BestMatch_Qn{qn}"] = best_text
                    row[f"{attr}_BestSim_Qn{qn}"] = round(float(best_sim), 4)
                    row[f"{attr}_Conf_Qn{qn}"] = round(float(conf), 4)

        # fill blanks consistently
        for attr in ORDERED_ATTRS:
            for qn in (1, 2, 3, 4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")
                if include_match_cols:
                    row.setdefault(f"{attr}_BestMatch_Qn{qn}", "")
                    row.setdefault(f"{attr}_BestSim_Qn{qn}", "")
                    row.setdefault(f"{attr}_Conf_Qn{qn}", "")

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
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # ordering
    ordered = [c for c in ["Date", "Duration", "Care_Staff"] if c in res.columns]
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date", "Duration", "Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    mid_q = []
    for attr in ORDERED_ATTRS:
        for qn in (1, 2, 3, 4):
            mid_q += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
            if include_match_cols:
                mid_q += [f"{attr}_BestSim_Qn{qn}", f"{attr}_Conf_Qn{qn}", f"{attr}_BestMatch_Qn{qn}"]
    ordered += [c for c in mid_q if c in res.columns]

    mid_a = []
    for attr in ORDERED_ATTRS:
        mid_a += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += [c for c in mid_a if c in res.columns]

    ordered += [c for c in ["Overall Total (0–21)"] if c in res.columns]
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
        <div class="header-card">
          <div class="pill">Thought Leadership • Option B (Meaning-only)</div>
          <h1>Thought Leadership Auto Scoring</h1>
          <p style="color:#6b7280;margin-top:0.3rem;">
            Scores answers by meaning: Answer text ↔ Exemplar text (per question_id).
            Top-K=40 (+ near-cutoff extras), temperature=1.0. No review flags.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load mapping + exemplars first (fast)
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

    with st.spinner("Preparing exemplar packs (per question_id)…"):
        packs_by_qid = build_packs_by_qid(exemplars)

    # Fetch Kobo
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

    # Score
    include_match_cols = st.sidebar.checkbox("Include BestMatch/Sim/Conf columns", value=True)
    with st.spinner("Scoring (Option B: Top-K vote)…"):
        scored = score_dataframe(df, mapping, packs_by_qid, include_match_cols=include_match_cols)

    st.success("✅ Done.")

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
