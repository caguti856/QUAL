# networking.py
# ------------------------------------------------------------
# Kobo → Exemplar-vote scoring (Option B++) + AI detection → Excel / Google Sheets
# Networking & Advocacy (A1..H1). Exact layout preserved.
# Batched embeddings + caching. AI-Suspect flag at end.
# Auto-run + auto-push options, stable widgets.
# ------------------------------------------------------------

from __future__ import annotations

import json, re, unicodedata
from dataclasses import dataclass
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
            --primary: #F26A21;
            --primary-soft: #FDE7D6;
            --gold: #FACC15;
            --gold-soft: #FEF9C3;
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

        h1 { font-size: 2.1rem; font-weight: 700; }
        p, span, label { color: var(--text-muted); }

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
            content: "";
            position: absolute;
            inset: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--gold-soft), var(--primary), var(--silver), var(--gold));
            opacity: 0.95;
        }
        .app-header-card::after {
            content: "";
            position: absolute;
            bottom: -40px;
            right: -40px;
            width: 140px;
            height: 140px;
            background: radial-gradient(circle, rgba(250,204,21,0.35), transparent 60%);
            opacity: 0.7;
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

        .stAlert { border-radius: 0.8rem; }

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

# mapping3.csv uses prefixes like SPD_Q#, PAS_Q#, etc.
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

QNUM_RX = re.compile(r"_Q(\d+)$")


# ==============================
# OPTION B++ KNOBS
# ==============================
TOPK_MAX    = int(st.secrets.get("TOPK_MAX", 30))
CLOSE_DELTA = float(st.secrets.get("CLOSE_DELTA", 0.08))
CLUSTER_SIM = float(st.secrets.get("CLUSTER_SIM", 0.78))
MIN_CLUSTER = int(st.secrets.get("MIN_CLUSTER", 6))
TEMP_CANDIDATES = [0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.30, 0.50, 1.00]


# ==============================
# AI DETECTION (aggressive)
# ==============================
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
        if qa_overlap(t, question_hint) < 0.06:
            score += 0.10

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
    for kind in ("submissions", "data"):
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
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {type(e).__name__}: {e}")
            return pd.DataFrame()

    st.error("Could not fetch data. Check KOBO_BASE, KOBO_ASSET_ID3, token permissions.")
    return pd.DataFrame()


# ==============================
# MAPPING + EXEMPLARS
# ==============================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns:
        m["prompt_hint"] = ""
    m["attribute"]   = m["attribute"].astype(str).map(clean)
    m["question_id"] = m["question_id"].astype(str).map(clean)
    m["prompt_hint"] = m["prompt_hint"].astype(str).map(clean)
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(","):
                s = s.lstrip(",").strip()
            rows.append(json.loads(s))
    return rows


# --------- Column resolver (keep your robust behavior) ---------
def build_bases_from_qid(question_id: str) -> list[str]:
    out = []
    if not question_id:
        return out
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m:
        return out
    qn = m.group(1)
    pref = qid.split("_Q")[0]
    sects = QID_PREFIX_TO_SECTIONS.get(pref, [])
    roots = ["networking", "Networking"]
    for sect in sects:
        for root in roots:
            out.append(f"{root}/{sect}_Section/{sect}_{qn}")
    return out

def expand_possible_kobo_columns(base: str) -> list[str]:
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
    c = col.lower(); t = token.lower()
    if c == t: return 100
    s = 0
    if c.endswith("/"+t): s = max(s,95)
    if f"/{t}/" in c: s = max(s,92)
    if f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c: s = max(s,90)
    if t in c: s = max(s,80)
    if "english" in c or "label" in c or "(en)" in c: s += 5
    if "answer (text)" in c or "answer_text" in c or "text" in c: s += 5
    if "networking/" in c: s += 2
    return s

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str, attribute: str = "") -> str | None:
    cols_original = list(df_cols)
    cols_lower = {c.lower(): c for c in cols_original}

    # 1) canonical path candidates from qid
    bases = build_bases_from_qid(question_id)
    for base in bases:
        lb = base.lower()
        if lb in cols_lower:
            return cols_lower[lb]
        for v in expand_possible_kobo_columns(base):
            lv = v.lower()
            if lv in cols_lower:
                return cols_lower[lv]
        for c in cols_original:
            if c.lower().startswith(lb):
                return c

    # 2) token candidates from qid + attribute section (A1..H1)
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

    for tok in tokens:
        lt = tok.lower()
        for c in cols_original:
            if lt in c.lower():
                return c

    best, bs = None, 0
    for tok in tokens:
        for c in cols_original:
            sc = _score_kobo_header(c, tok)
            if sc > bs:
                bs, best = sc, c
    if best and bs >= 80:
        return best

    # 3) fuzzy rescue by prompt
    hint = clean(prompt_hint or "")
    if hint:
        cands = [(c, c.lower()) for c in cols_original]
        hits = process.extract(hint.lower(), [lo for _, lo in cands], scorer=fuzz.partial_token_set_ratio, limit=6)
        for _, lo, score in hits:
            if score >= 88:
                for orig, low in cands:
                    if low == lo:
                        return orig

    return None


# ==============================
# EMBEDDINGS + EXEMPLAR PACKS (Option B++)
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(st.secrets.get("EMBED_MODEL", "all-MiniLM-L6-v2"))

@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: tuple[str, ...]) -> np.ndarray:
    model = get_embedder()
    embs = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64
    ).astype(np.float32)
    return embs

def embed_map(texts: list[str]) -> dict[str, np.ndarray]:
    uniq, seen = [], set()
    for t in texts:
        t = clean(t)
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return {}
    embs = embed_texts_cached(tuple(uniq))
    return {t: v for t, v in zip(uniq, embs)}

@dataclass
class ExemplarPack:
    vecs: np.ndarray
    scores: np.ndarray
    texts: list[str]

def build_packs_by_question(exemplars: list[dict]) -> dict[str, ExemplarPack]:
    by_qid: dict[str, dict[str, list]] = {}
    all_texts: list[str] = []

    for e in exemplars:
        qid = clean(e.get("question_id",""))
        txt = clean(e.get("text",""))
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

    packs: dict[str, ExemplarPack] = {}
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
                texts=texts
            )
    return packs

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

def vote_with_temp(pack: ExemplarPack, idx: np.ndarray, sims: np.ndarray, temp: float) -> tuple[int, float, float]:
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

def score_answer_auto(pack: ExemplarPack | None, ans_vec: np.ndarray | None) -> int:
    if pack is None or pack.vecs.size == 0 or ans_vec is None:
        return 1

    sims = (pack.vecs @ ans_vec).astype(np.float32)
    if sims.size == 0:
        return 1

    top_idx = _topk_sorted(sims, k=min(TOPK_MAX, sims.size))
    thematic_idx = select_thematic_subset(pack, top_idx, sims)

    best = None  # (margin, conf, pred)
    for t in TEMP_CANDIDATES:
        pred, conf, margin = vote_with_temp(pack, thematic_idx, sims, temp=float(t))
        cand = (margin, conf, pred)
        if best is None or cand[0] > best[0] + 1e-9 or (abs(cand[0]-best[0]) < 1e-9 and cand[1] > best[1] + 1e-9):
            best = cand
    return int(best[2]) if best else 1


# ==============================
# SCORING (layout preserved, Q1..Q4)
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame, packs_by_qid: dict[str, ExemplarPack]) -> pd.DataFrame:
    df_cols = list(df.columns)

    with st.expander("🔎 Networking-like columns found", expanded=False):
        sample_cols = [c for c in df_cols if "/A" in c or "/B" in c or "/C" in c or "/D" in c or "/E" in c or "/F" in c or "/G" in c or "/H" in c or "Networking/" in c]
        st.write(sample_cols[:120])

    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id","staff_id","staffid")), None)
    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end"] if c in df.columns), None)

    n_rows = len(df)

    dt_series = (
        pd.to_datetime(df[date_col].astype(str).str.strip().str.lstrip(","), errors="coerce")
        if date_col in df.columns else pd.Series([pd.NaT] * n_rows)
    )

    start_dt = pd.to_datetime(df[start_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce") if start_col else pd.Series([pd.NaT]*n_rows)
    end_dt   = pd.to_datetime(df[end_col].astype(str).str.strip().str.lstrip(","), utc=True, errors="coerce") if end_col else pd.Series([pd.NaT]*n_rows)

    duration_min = ((end_dt - start_dt).dt.total_seconds() / 60.0).clip(lower=0)

    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

    # resolve Kobo headers
    resolved_for_qid = {}
    missing = []
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        attr  = r.get("attribute","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint, attr)
        if hit:
            resolved_for_qid[qid] = hit
        else:
            missing.append((qid, attr, qhint))

    with st.expander("🧭 Mapping → Kobo column resolution (by question_id)", expanded=False):
        if resolved_for_qid:
            show = [(k, v) for k, v in list(resolved_for_qid.items())[:120]]
            st.dataframe(pd.DataFrame(show, columns=["question_id","kobo_column"]))
        if missing:
            st.warning(f"{len(missing)} mapping rows not found in headers (showing up to 40).")
            st.dataframe(pd.DataFrame(missing[:40], columns=["question_id","attribute","prompt_hint"]))

    # batch embed all unique answers
    distinct_answers = set()
    for _, row in df.iterrows():
        for r in all_mapping:
            col = resolved_for_qid.get(r["question_id"])
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a:
                    distinct_answers.add(a)

    ans_emb = embed_map(list(distinct_answers))

    rows_out = []
    for i, resp in df.iterrows():
        out = {}
        out["Date"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_series.iloc[i]) else str(i)
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        d_val = duration_min.iloc[i]
        out["Duration_min"] = int(round(d_val)) if not pd.isna(d_val) else ""

        per_attr: dict[str, list[int]] = {}
        ai_scores: list[float] = []

        for r in all_mapping:
            qid   = r["question_id"]
            attr  = r["attribute"]
            qhint = r.get("prompt_hint","")

            # only Q1..Q4
            qn = None
            if "_Q" in (qid or ""):
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None
            if qn not in (1,2,3,4):
                continue

            col = resolved_for_qid.get(qid)
            if not col or col not in df.columns:
                continue

            ans = clean(resp.get(col, ""))
            if not ans:
                continue

            ai_scores.append(ai_signal_score(ans, qhint))

            vec = ans_emb.get(ans)
            pack = packs_by_qid.get(clean(qid), None)
            sc = score_answer_auto(pack, vec)

            # light off-topic guard
            if qa_overlap(ans, qhint) < MIN_QA_OVERLAP:
                sc = min(sc, 1)

            out[f"{attr}_Qn{qn}"] = int(sc)
            out[f"{attr}_Rubric_Qn{qn}"] = BANDS[int(sc)]
            per_attr.setdefault(attr, []).append(int(sc))

        # fixed shape for blocks
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + ranks + overall
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                out[f"{attr}_Avg (0–3)"] = ""
                out[f"{attr}_RANK"] = ""
            else:
                avg = float(np.mean(scores))
                band = int(np.clip(int(round(avg)), 0, 3))
                overall_total += band
                out[f"{attr}_Avg (0–3)"] = round(avg, 2)
                out[f"{attr}_RANK"] = BANDS[band]

        out["Overall Total (0–24)"] = overall_total
        out["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")

        out["AI_MaxScore"]  = round(float(max(ai_scores) if ai_scores else 0.0), 2)
        out["AI-Suspected"] = bool(any(s >= AI_SUSPECT_THRESHOLD for s in ai_scores))

        rows_out.append(out)

    res_df = pd.DataFrame(rows_out)

    def order_cols(cols):
        ordered = ["Date","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI_MaxScore", "AI-Suspected"]
        extras = [c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols] + extras

    return res_df.reindex(columns=order_cols(list(res_df.columns)))


# ==============================
# EXPORTS / SHEETS
# ==============================
def _ensure_ai_last(df: pd.DataFrame, export_name: str = "AI-Suspected") -> pd.DataFrame:
    out = df.copy()
    if export_name not in out.columns:
        out[export_name] = ""
    cols = [c for c in out.columns if c not in ("AI_MaxScore", export_name)] + ["AI_MaxScore", export_name]
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
    sh = gc.open_by_key(key)
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
    except Exception:
        pass

def upload_df_to_gsheets(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        df = _ensure_ai_last(df)
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
# PAGE ENTRYPOINT
# ==============================
def main():
    inject_css()

    st.markdown("""
        <div class="app-header-card">
            <div class="pill">Networking & Advocacy • Option B++</div>
            <h1>Networking & Advocacy</h1>
            <p class="app-header-subtitle">
                Kobo → exemplar-vote scoring (per question_id) + AI detection → Excel / Google Sheets.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.session_state.setdefault("scored_df_networking", None)
    st.session_state.setdefault("excel_bytes_networking", b"")

    def run_pipeline():
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

        with st.spinner("Building exemplar packs (cached)…"):
            packs_by_qid = build_packs_by_question(exemplars)

        with st.spinner("Fetching Kobo submissions…"):
            df = fetch_kobo_dataframe()
        if df.empty:
            st.warning("No Kobo submissions found.")
            return

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📥 Fetched dataset")
        st.caption(f"Rows: {len(df):,}  •  Columns: {len(df.columns):,}")
        st.dataframe(df, use_container_width=True, height=360)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Scoring (Option B++) + AI detection…"):
            scored_df = score_dataframe(df, mapping, packs_by_qid)

        st.success("✅ Scoring complete.")

        def _highlight_ai(row):
            if "AI-Suspected" in row and row["AI-Suspected"]:
                return ["background-color: #241E4E"] * len(row)
            return [""] * len(row)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Scored table")
        st.dataframe(scored_df.style.apply(_highlight_ai, axis=1), use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state["scored_df_networking"] = scored_df
        st.session_state["excel_bytes_networking"] = to_excel_bytes(scored_df)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⬇️ Export")
        st.download_button(
            "Download Excel",
            data=st.session_state["excel_bytes_networking"],
            file_name="Networking_Scoring.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_xlsx_networking",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if AUTO_PUSH:
            with st.spinner("📤 Sending to Google Sheets..."):
                ok, msg = upload_df_to_gsheets(scored_df)
            (st.success if ok else st.error)(msg)

    # Auto-run once per session
    if AUTO_RUN and not st.session_state.get("networking_auto_ran_once"):
        st.session_state["networking_auto_ran_once"] = True
        run_pipeline()

    if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True, key="btn_run_networking"):
        run_pipeline()

    if (st.session_state.get("scored_df_networking") is not None) and (not AUTO_PUSH):
        with st.expander("Google Sheets export", expanded=True):
            st.write("Spreadsheet key:", st.secrets.get("GSHEETS_SPREADSHEET_KEY") or "⚠️ Not set")
            st.write("Worksheet name:", DEFAULT_WS_NAME)
            if st.button("📤 Send scored table to Google Sheets", use_container_width=True, key="btn_push_networking"):
                ok, msg = upload_df_to_gsheets(st.session_state["scored_df_networking"])
                (st.success if ok else st.error)(msg)

if __name__ == "__main__":
    main()
