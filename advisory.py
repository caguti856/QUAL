# advisory.py — Kobo → Exemplar-vote scoring (Option B++) + per-answer AI detection → Excel / Google Sheets
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
# UI / STYLING (same style)
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
KOBO_BASE     = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID = st.secrets.get("KOBO_ASSET_ID", "")
KOBO_TOKEN    = st.secrets.get("KOBO_TOKEN", "")
AUTO_RUN      = bool(st.secrets.get("AUTO_RUN", False))
AUTO_PUSH     = bool(st.secrets.get("AUTO_PUSH", False))

DATASETS_DIR   = Path("DATASETS")
MAPPING_PATH   = DATASETS_DIR / "mapping.csv"
EXEMPLARS_PATH = DATASETS_DIR / "advisory_exemplars_smart.cleaned.jsonl"



# =============================================================================
# CONSTANTS
# =============================================================================
BANDS = {0: "Counterproductive", 1: "Compliant", 2: "Strategic", 3: "Transformative"}

ORDERED_ATTRS = [
    "Strategic & analytical thinking",
    "Credibility & trustworthiness",
    "Effective communication & influence",
    "Client & stakeholder focus",
    "Fostering collaboration & partnership",
    "Ensuring relevance & impact",
    "Solution orientation & adaptability",
    "Capacity strengthening & empowerment support",
]

OVERALL_BANDS = [
    ("Exemplary Thought Leader", 21, 24),
    ("Strategic Advisor",       16, 20),
    ("Emerging Advisor",        10, 15),
    ("Needs Capacity Support",   0,  9),
]

# QID prefix → Kobo section (your advisory form)
QID_PREFIX_TO_SECTION = {
    "SAT": "A1",
    "CT":  "A2",
    "ECI": "A3",
    "CSF": "A4",
    "FCP": "A5",
    "ERI": "A6",
    "SOA": "A7",
    "CSE": "A8",
}

# Scoring defaults (automatic) — same philosophy as Leadership Option B++
TOPK_MAX    = int(st.secrets.get("TOPK_MAX", 30))
CLOSE_DELTA = float(st.secrets.get("CLOSE_DELTA", 0.08))
CLUSTER_SIM = float(st.secrets.get("CLUSTER_SIM", 0.78))
MIN_CLUSTER = int(st.secrets.get("MIN_CLUSTER", 6))

TEMP_CANDIDATES = [0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.30, 0.50, 1.00]


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
# AI DETECTION (same as Leadership)
# =============================================================================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

TRANSITION_OPEN_RX = re.compile(
    r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b",
    re.I
)
LIST_CUES_RX       = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX          = re.compile(r"^[-*•]\s", re.M)
LONG_DASH_HARD_RX  = re.compile(r"[—–]")
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
DAY_RANGE_RX        = re.compile(r"\bday\s*\d+\s*[-–]\s*\d+\b", re.I)
PIPE_LIST_RX        = re.compile(r"\s\|\s")
PARENS_ACRONYMS_RX  = re.compile(r"\(([A-Z]{2,}(?:s)?(?:\s*,\s*[A-Z]{2,}(?:s)?)+).*?\)")
NUMBERED_BULLETS_RX = re.compile(r"\b\d+\s*[\.\)]\s*")
SLASH_PAIR_RX       = re.compile(r"\b\w+/\w+\b")


def qa_overlap(ans, qtext) -> float:
    def _t(x) -> str:
        if x is None:
            return ""
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
    t = clean(text)
    if not t:
        return 0.0
    if LONG_DASH_HARD_RX.search(t):
        return 1.0

    score = 0.0
    if SYMBOL_RX.search(t):               score += 0.35
    if TIMEBOX_RX.search(t):              score += 0.15
    if AI_RX.search(t):                   score += 0.35
    if TRANSITION_OPEN_RX.search(t):      score += 0.12
    if LIST_CUES_RX.search(t):            score += 0.12
    if BULLET_RX.search(t):               score += 0.08

    if DAY_RANGE_RX.search(t):            score += 0.15
    if PIPE_LIST_RX.search(t):            score += 0.10
    if PARENS_ACRONYMS_RX.search(t):      score += 0.10
    if NUMBERED_BULLETS_RX.search(t):     score += 0.12
    if SLASH_PAIR_RX.search(t):           score += 0.08

    hits = 0
    for rx in (TIMEBOX_RX, DAY_RANGE_RX, PIPE_LIST_RX, NUMBERED_BULLETS_RX):
        if rx.search(t):
            hits += 1
    if hits >= 2: score += 0.25
    if hits >= 3: score += 0.15

    if question_hint:
        overlap = qa_overlap(t, question_hint)
        if overlap < 0.06:
            score += 0.10

    return max(0.0, min(1.0, score))


# =============================================================================
# KOBO
# =============================================================================
def kobo_url(asset_uid: str, kind: str = "submissions") -> str:
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()

    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions", "data"):
        url = kobo_url(KOBO_ASSET_ID, kind)
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

    # expected: column, question_id, attribute, prompt_hint (prompt_hint optional)
    if "prompt_hint" not in df.columns:
        df["prompt_hint"] = ""

    required = {"question_id", "attribute"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"mapping.csv must include: {', '.join(sorted(required))}")

    # keep only known attributes
    df["attribute"] = df["attribute"].astype(str).map(clean)
    df = df[df["attribute"].isin(ORDERED_ATTRS)].copy()

    df["question_id"] = df["question_id"].astype(str).map(clean)
    df["prompt_hint"] = df["prompt_hint"].astype(str).map(clean)
    return df


QNUM_RX = re.compile(r"_Q(\d+)$")


def build_kobo_base_from_qid(question_id: str) -> Optional[str]:
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
    return f"Advisory/{sect}_Section/{sect}_{qn}"


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
    c = clean(col).lower()
    t = clean(token).lower()
    if not c or not t:
        return 0
    if c == t:
        return 100
    if t in c:
        return max(80, int(fuzz.token_set_ratio(t, c)))
    return int(fuzz.token_set_ratio(t, c))


def resolve_kobo_column_for_mapping(
    df_cols: List[str],
    question_id: str,
    prompt_hint: str,
    mapping_column_text: str = "",
) -> Optional[str]:
    # 1) canonical base + variants
    base = build_kobo_base_from_qid(question_id)
    if base and base in df_cols:
        return base
    if base:
        for v in expand_possible_kobo_columns(base):
            if v in df_cols:
                return v
        for c in df_cols:
            if c.startswith(base):
                return c

    # 2) token match (A1_#, A2_# etc)
    token = None
    qid = (question_id or "").strip().upper()
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

    # 3) fuzzy prompt_hint rescue
    hint = clean(prompt_hint or "")
    if hint:
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=6)
        for col, score, _ in hits:
            if score >= 88:
                return col

    # 4) fuzzy mapping['column'] text rescue
    mct = clean(mapping_column_text or "")
    if mct:
        hits = process.extract(mct, df_cols, scorer=fuzz.partial_token_set_ratio, limit=6)
        for col, score, _ in hits:
            if score >= 88:
                return col

    return None


# =============================================================================
# EXEMPLARS (JSONL)
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
            packs[qid] = ExemplarPack(
                vecs=np.zeros((0, 384), dtype=np.float32),
                scores=np.array([], dtype=np.int32),
                texts=[]
            )
        else:
            packs[qid] = ExemplarPack(
                vecs=np.vstack(vecs).astype(np.float32),
                scores=np.array(scores, dtype=np.int32),
                texts=texts,
            )
    return packs


# =============================================================================
# OPTION B++: MEANING SELECTION + PER-ANSWER TEMP SELECTION
# =============================================================================
def _softmax_temp(x: np.ndarray, temp: float) -> np.ndarray:
    t = float(max(1e-6, temp))
    z = (x - float(x.max())) / t
    ex = np.exp(z).astype(np.float32)
    return ex / (ex.sum() + 1e-9)


def _topk_sorted(sims: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, sims.size)))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx


def select_thematic_subset(pack: ExemplarPack, top_idx: np.ndarray, sims: np.ndarray) -> np.ndarray:
    if top_idx.size == 0:
        return top_idx

    best_i = int(top_idx[0])
    best_sim = float(sims[best_i])

    close = [i for i in top_idx.tolist() if float(sims[i]) >= best_sim - CLOSE_DELTA]
    if len(close) < MIN_CLUSTER:
        close = top_idx[:max(MIN_CLUSTER, min(12, top_idx.size))].tolist()

    seed_vec = pack.vecs[best_i]
    close_vecs = pack.vecs[np.array(close, dtype=np.int64)]
    sim_to_seed = (close_vecs @ seed_vec).astype(np.float32)

    thematic = [i for i, s in zip(close, sim_to_seed.tolist()) if s >= CLUSTER_SIM]
    if len(thematic) < MIN_CLUSTER:
        thematic = close

    return np.array(thematic, dtype=np.int64)


def vote_with_temp(pack: ExemplarPack, idx: np.ndarray, sims: np.ndarray, temp: float) -> Tuple[int, float, float]:
    if idx.size == 0:
        return 1, 0.0, 0.0

    top_sims = sims[idx].astype(np.float32)
    top_scores = pack.scores[idx].astype(np.int32)

    w = _softmax_temp(top_sims, temp=temp)

    class_w = np.zeros(4, dtype=np.float32)
    for sc, wi in zip(top_scores.tolist(), w.tolist()):
        if 0 <= int(sc) <= 3:
            class_w[int(sc)] += float(wi)

    pred = int(class_w.argmax())
    conf = float(class_w.max())

    sorted_w = np.sort(class_w)
    second = float(sorted_w[-2]) if sorted_w.size >= 2 else 0.0
    margin = conf - second
    return pred, conf, margin


def score_answer_auto(pack: ExemplarPack, ans_vec: np.ndarray) -> Optional[int]:
    if pack is None or pack.vecs.size == 0 or ans_vec is None:
        return None

    sims = (pack.vecs @ ans_vec).astype(np.float32)
    if sims.size == 0:
        return None

    top_idx = _topk_sorted(sims, k=min(TOPK_MAX, sims.size))
    thematic_idx = select_thematic_subset(pack, top_idx, sims)

    best = None  # (margin, conf, pred)
    for t in TEMP_CANDIDATES:
        pred, conf, margin = vote_with_temp(pack, thematic_idx, sims, temp=float(t))
        cand = (margin, conf, pred)
        if best is None or cand[0] > best[0] + 1e-9 or (abs(cand[0] - best[0]) < 1e-9 and cand[1] > best[1] + 1e-9):
            best = cand

    if best is None:
        return None
    return int(best[2])


# =============================================================================
# EXPORT
# =============================================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return bio.getvalue()


def _ensure_ai_last(df: pd.DataFrame, export_name: str = "AI_Suspected", source_name: str = "AI_suspected") -> pd.DataFrame:
    out = df.copy()
    if export_name not in out.columns:
        if source_name in out.columns:
            out = out.rename(columns={source_name: export_name})
        else:
            out[export_name] = ""
    cols = [c for c in out.columns if c != export_name] + [export_name]
    return out[cols]


# =============================================================================
# GOOGLE SHEETS
# =============================================================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME", "Advisory")


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
        df_out = _ensure_ai_last(df, export_name="AI_Suspected", source_name="AI_suspected")
        header = df_out.columns.astype(str).tolist()
        values = df_out.astype(object).where(pd.notna(df_out), "").values.tolist()
        ws.clear()
        ws.spreadsheet.values_batch_update(
            body={"valueInputOption": "USER_ENTERED", "data": [{"range": f"'{ws.title}'!A1", "values": [header] + values}]}
        )
        return True, f"✅ Wrote {len(values)} rows × {len(header)} cols to '{ws.title}'."
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"


# =============================================================================
# SCORING + SHADING (same idea)
# =============================================================================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame, packs_by_qid: Dict[str, ExemplarPack]) -> pd.DataFrame:
    df_cols = list(df.columns)

    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id", "staff_id", "staffid")), None)

    date_cols_pref = ["_submission_time", "SubmissionDate", "submissiondate", "end", "End", "start", "Start", "today", "date", "Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start"] if c in df.columns), None)
    end_col   = next((c for c in ["end"] if c in df.columns), None)

    n_rows = len(df)

    dt_series = (
        pd.to_datetime(df[date_col].astype(str).str.strip().str.lstrip(","), errors="coerce")
        if date_col in df.columns else pd.Series([pd.NaT] * n_rows)
    )

    if start_col:
        start_dt = pd.to_datetime(df[start_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)

    if end_col:
        end_dt = pd.to_datetime(df[end_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT] * n_rows)

    duration_min = ((end_dt - start_dt).dt.total_seconds() / 60.0).clip(lower=0)

    rows = mapping.to_dict(orient="records")

    # resolve Kobo columns per question_id
    resolved_for_qid: Dict[str, str] = {}
    missing_qids = []

    for r in rows:
        qid = clean(r.get("question_id", ""))
        qhint = r.get("prompt_hint", "") or ""
        mcol = r.get("column", "") if "column" in r else ""
        col = resolve_kobo_column_for_mapping(df_cols, qid, qhint, mcol)
        if col:
            resolved_for_qid[qid] = col
        else:
            missing_qids.append(qid)

    # Batch-embed ALL unique answers
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
    exact_cache: Dict[Tuple[str, str], int] = {}

    out_rows = []
    for i, rec in df.iterrows():
        row = {}
        row["Date"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_series.iloc[i]) else str(i)
        row["Duration_min"] = int(round(duration_min.iloc[i])) if not pd.isna(duration_min.iloc[i]) else ""
        row["Staff ID"] = str(rec.get(staff_id_col)) if staff_id_col else ""

        per_attr: Dict[str, List[int]] = {}
        any_ai = False

        for r in rows:
            qid = clean(r.get("question_id", ""))
            attr = clean(r.get("attribute", ""))
            qhint = clean(r.get("prompt_hint", ""))

            if attr not in ORDERED_ATTRS:
                continue

            col = resolved_for_qid.get(qid)
            if not col or col not in df.columns:
                continue

            # only score Q1..Q4 like Leadership style
            qn = None
            if "_Q" in qid:
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None
            if qn not in (1, 2, 3, 4):
                continue

            ans = clean(rec.get(col, ""))
            if not ans:
                continue

            # AI detection
            if ai_signal_score(ans, qhint) >= AI_SUSPECT_THRESHOLD:
                any_ai = True

            cache_key = (qid, ans)
            if cache_key in exact_cache:
                sc = exact_cache[cache_key]
            else:
                vec = ans_emb.get(ans)
                pack = packs_by_qid.get(qid)
                sc2 = score_answer_auto(pack, vec)
                if sc2 is None:
                    sc2 = 1
                sc = int(sc2)
                exact_cache[cache_key] = sc

            row[f"{attr}_Qn{qn}"] = sc
            row[f"{attr}_Rubric_Qn{qn}"] = BANDS[int(sc)]
            per_attr.setdefault(attr, []).append(int(sc))

        # ensure blank columns exist consistently
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
                band = int(np.clip(int(round(avg)), 0, 3))
                overall_total += band
                row[f"{attr}_Avg (0–3)"] = round(avg, 2)
                row[f"{attr}_RANK"] = BANDS[band]

        row["Overall Total (0–24)"] = overall_total
        row["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        row["AI_suspected"] = bool(any_ai)

        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # order columns
    ordered = ["Date", "Duration_min", "Staff ID"]
    q_cols = []
    for attr in ORDERED_ATTRS:
        for qn in (1, 2, 3, 4):
            q_cols += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
    ordered += q_cols

    a_cols = []
    for attr in ORDERED_ATTRS:
        a_cols += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += a_cols

    ordered += ["Overall Total (0–24)", "Overall Rank", "AI_suspected"]
    res = res.reindex(columns=[c for c in ordered if c in res.columns])

    if missing_qids:
        st.session_state["missing_qids"] = sorted(set([m for m in missing_qids if m]))

    return res


def style_ai_rows(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def _row_style(r):
        try:
            flagged = ("AI_suspected" in r and bool(r["AI_suspected"]))
        except Exception:
            flagged = False
        if flagged:
            return ["background-color: #241E4E"] * len(r)
        return [""] * len(r)
    return df.style.apply(_row_style, axis=1)


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()

    st.markdown(
        """
        <div class="app-header-card">
            <div class="pill">Advisory • Scoring and AI Detection</div>
            <h1>Advisory Scoring</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def run():
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
        st.dataframe(df, use_container_width=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Scoring (Auto meaning vote per question + AI detection)…"):
            scored = score_dataframe(df=df, mapping=mapping, packs_by_qid=packs_by_qid)

        missing_qids = st.session_state.get("missing_qids", [])
        if missing_qids:
            st.info(
                "Some mapping question_id(s) could not be matched to Kobo columns: "
                + ", ".join(missing_qids[:12])
                + (" …" if len(missing_qids) > 12 else "")
            )

        ai_ct = int(scored["AI_suspected"].fillna(False).astype(bool).sum()) if "AI_suspected" in scored.columns else 0
        st.success(f"✅ Scoring complete. AI flagged: {ai_ct:,} row(s).")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Scored table (AI rows shaded)")
        st.dataframe(style_ai_rows(scored), use_container_width=True, height=420)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⬇️ Export")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Excel",
                data=to_excel_bytes(scored),
                file_name="Advisory_Scored.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download CSV",
                data=_ensure_ai_last(scored, export_name="AI_Suspected", source_name="AI_suspected").to_csv(index=False).encode("utf-8"),
                file_name="Advisory_Scored.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if AUTO_PUSH:
            with st.spinner("📤 Sending to Google Sheets…"):
                ok, msg = upload_df_to_gsheets(scored)
            (st.success if ok else st.error)(msg)

        st.session_state["scored_df"] = scored

    # auto-run once if configured
    if AUTO_RUN and not st.session_state.get("advisory_auto_ran_once"):
        st.session_state["advisory_auto_ran_once"] = True
        run()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True):
        run()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
