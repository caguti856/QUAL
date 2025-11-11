# file: leadership.py — Thought Leadership Scoring (auto-run + auto-push, stronger AI detection)

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

import json, re, unicodedata, glob, os
from io import BytesIO
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# ==============================
# CONSTANTS / PATHS / SECRETS
# ==============================
KOBO_BASE       = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID   = st.secrets.get("KOBO_ASSET_ID1", "")
KOBO_TOKEN      = st.secrets.get("KOBO_TOKEN", "")

AUTO_RUN        = True   # force auto-run
AUTO_PUSH       = True   # force auto-push

DATASETS_DIR    = Path("DATASETS")
MAPPING_PATH    = Path(st.secrets.get("LEADERSHIP_MAPPING_PATH", DATASETS_DIR / "mapping1.csv"))
EXEMPLARS_PATH  = Path(st.secrets.get("LEADERSHIP_EXEMPLARS_PATH", DATASETS_DIR / "thought_leadership.cleaned.jsonl"))

# Fallback local CSV (for offline/testing)
FALLBACK_GLOB   = st.secrets.get("FALLBACK_EXPORT_GLOB", "/mnt/data/*export.csv")

BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 21, 24),
    ("Strategic Advisor",        16, 20),
    ("Emerging Advisor",         10, 15),
    ("Needs Capacity Support",    0,  9),
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

# ---- AI detection (richer) ----
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.50))  # stricter default
TRANSITION_OPEN_RX = re.compile(r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b", re.I)
LIST_CUES_RX       = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX          = re.compile(r"(?m)^\s*[-*•]\s+")
NUMBERED_LIST_RX   = re.compile(r"(?m)^\s*(?:\d+\.|\(\d+\))\s+")
MATH_SYMBOLS_RX    = re.compile(r"[≤≥≠±×÷→←↔≈§†‡…]{1,}")
CLAUSE_CHAIN_RX    = re.compile(r"(?:;|—|–).*(?:;|—|–)")
TIMEBOX_RX         = re.compile(r"\b(?:daily|weekly|biweekly|fortnightly|monthly|quarterly|sprint|cadence)\b", re.I)
TEMPLATE_RX        = re.compile(r"\b(?:as an ai|i am an ai|as a language model)\b", re.I)
LINKY_RX           = re.compile(r"https?://|www\.", re.I)
QUANT_RX           = re.compile(r"\b\d+(\.\d+)?\s*(?:%|percent|pct|/10|/100|kpi|okrs?)\b", re.I)
QUOTE_HEAVY_RX     = re.compile(r"[“”\"'].*[“”\"']")  # nested quotes
TITLE_CASE_HEAVY_RX= re.compile(r"(?:\b[A-Z][a-z]+(?:\s+|$)){8,}")  # long proper-title runs

# ==============================
# HELPERS
# ==============================
def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def try_dt(x):
    if pd.isna(x): return None
    if isinstance(x, (pd.Timestamp, datetime)): return pd.to_datetime(x)
    try: return pd.to_datetime(str(x), errors="coerce")
    except Exception: return None

def cos_sim(a, b):
    if a is None or b is None: return -1e9
    return float(np.dot(a, b))

def qa_overlap(ans: str, qtext: str) -> float:
    at = set(re.findall(r"\w+", (ans or "").lower()))
    qt = set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

def _avg_sentence_len(text: str) -> float:
    s = re.split(r"[.!?]+", text or "")
    s = [w for w in s if w.strip()]
    if not s: return 0.0
    tokens = re.findall(r"\w+", text or "")
    return len(tokens) / max(len(s), 1)

def _type_token_ratio(text: str) -> float:
    toks = [t.lower() for t in re.findall(r"[a-z]+", text or "")]
    if not toks: return 1.0
    return len(set(toks)) / len(toks)

def ai_signal_score(text: str, question_hint: str = "") -> tuple[float, list[str]]:
    """Heuristic AI-ishness in [0,1]."""
    t = clean(text)
    flags = []
    if not t:
        return 0.0, flags

    score = 0.0

    # Structure/templates
    if TRANSITION_OPEN_RX.search(t): score += 0.12; flags.append("style:transition-opening")
    if LIST_CUES_RX.search(t):       score += 0.10; flags.append("style:list-cues")
    if BULLET_RX.search(t):          score += 0.08; flags.append("format:bullets")
    if NUMBERED_LIST_RX.search(t):   score += 0.08; flags.append("format:numbered-list")
    if CLAUSE_CHAIN_RX.search(t):    score += 0.07; flags.append("syntax:clause-chains")

    # Lexical/symbolic
    if MATH_SYMBOLS_RX.search(t):    score += 0.08; flags.append("lex:math-symbols")
    if TIMEBOX_RX.search(t):         score += 0.05; flags.append("lex:timeboxing")
    if LINKY_RX.search(t):           score += 0.04; flags.append("format:links")
    if QUANT_RX.search(t):           score += 0.04; flags.append("lex:quantified")
    if QUOTE_HEAVY_RX.search(t):     score += 0.03; flags.append("style:quote-heavy")
    if TITLE_CASE_HEAVY_RX.search(t):score += 0.04; flags.append("style:titlecase-runs")
    if TEMPLATE_RX.search(t):        score += 0.35; flags.append("pattern:ai-boilerplate")

    # Sentence/variety stats
    asl = _avg_sentence_len(t)
    if asl >= 26:                     score += 0.16; flags.append(f"syntax:long-sentences(~{int(asl)})")
    elif asl >= 18:                   score += 0.10; flags.append(f"syntax:moderate-long(~{int(asl)})")
    ttr = _type_token_ratio(t)
    if ttr <= 0.45 and len(t) >= 180: score += 0.10; flags.append(f"lex:low-variety(ttr={ttr:.2f})")

    # Q/A overlap guard
    if question_hint:
        overlap = qa_overlap(t, question_hint)
        if overlap < 0.06:
            score += 0.10; flags.append(f"qa:low-overlap({overlap:.2f})")

    score = max(0.0, min(1.0, score))
    return score, flags

def show_status(ok: bool, msg: str) -> None:
    (st.success if ok else st.error)(msg)

def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

# ==============================
# LOADERS
# ==============================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        # try /mnt/data fallback
        alt = Path("/mnt/data/mapping1.csv")
        if alt.exists(): path = alt
        else: raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    # snap attribute names
    norm = lambda s: re.sub(r"\s+"," ", str(s).strip().lower())
    target = {norm(a): a for a in ORDERED_ATTRS}
    def snap_attr(a):
        key = norm(a)
        if key in target: return target[key]
        best = process.extractOne(key, list(target.keys()), scorer=fuzz.token_set_ratio)
        return target[best[0]] if best and best[1] >= 72 else None
    m["attribute"] = m["attribute"].apply(snap_attr)
    m = m[m["attribute"].notna()].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists():
        alt = Path("/mnt/data/thought_leadership.cleaned.jsonl")
        if alt.exists(): path = alt
        else: raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if KOBO_ASSET_ID and KOBO_TOKEN:
        headers = {"Authorization": f"Token {KOBO_TOKEN}"}
        for kind in ("submissions","data"):
            url = kobo_url(KOBO_ASSET_ID, kind)
            try:
                r = requests.get(url, headers=headers, timeout=60)
                if r.status_code == 404: continue
                r.raise_for_status()
                payload = r.json()
                results = payload if isinstance(payload, list) else payload.get("results", [])
                if not results and "results" not in payload: results = payload
                df = pd.DataFrame(results)
                if not df.empty: df.columns = [str(c).strip() for c in df.columns]
                if not df.empty: return df
            except Exception:
                pass  # fall back
    # fallback: local CSV export
    candidates = sorted(glob.glob(FALLBACK_GLOB), key=os.path.getmtime, reverse=True)
    if candidates:
        try:
            df = pd.read_csv(candidates[0])
            df.columns = [str(c).strip() for c in df.columns]
            st.info(f"Used local export: {os.path.basename(candidates[0])}")
            return df
        except Exception:
            pass
    st.warning("No Kobo data and no local export found.")
    return pd.DataFrame()

# ==============================
# QUESTION_ID → KOBO COLUMN RESOLVER
# ==============================
QID_PREFIX_TO_SECTION = {
    "LAV": "A1",
    "II":  "A2",
    "EP":  "A3",
    "CFC": "A4",
    "FTD": "A5",
    "LDA": "A6",
    "RDM": "A7",
}
QNUM_RX = re.compile(r"_Q(\d+)$")

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
        if v in df_cols:
            return v
    for c in df_cols:
        if any(c.startswith(b) for b in bases):
            return c
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
            if score >= 80:
                return col
    return None

# ==============================
# EMBEDDINGS / CENTROIDS
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_centroids(exemplars: list[dict]):
    by_qkey, by_attr, question_texts = {}, {}, []
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

    embedder = get_embedder()

    def centroid(texts):
        if not texts: return None
        embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embs.mean(axis=0)

    def build_centroids_for_q(texts, scores):
        buckets = {0:[],1:[],2:[],3:[]}
        for t,s in zip(texts, scores):
            if t: buckets[int(s)].append(t)
        return {sc: centroid(batch) for sc, batch in buckets.items()}

    q_centroids     = {k: build_centroids_for_q(v["texts"], v["scores"]) for k, v in by_qkey.items()}
    attr_centroids  = {attr: {sc: centroid(txts) for sc, txts in bucks.items()} for attr, bucks in by_attr.items()}
    global_buckets  = {0:[],1:[],2:[],3:[]}
    for e in exemplars:
        sc = int(e.get("score",0)); txt = clean(e.get("text",""))
        if txt: global_buckets[sc].append(txt)
    global_centroids= {sc: centroid(txts) for sc, txts in global_buckets.items()}

    return q_centroids, attr_centroids, global_centroids, by_qkey, question_texts

_embed_cache: dict[str, np.ndarray] = {}
def embed_cached(text: str):
    t = clean(text)
    if not t: return None
    if t in _embed_cache: return _embed_cache[t]
    vec = get_embedder().encode(t, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    _embed_cache[t] = vec
    return vec

def resolve_qkey(q_centroids, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in q_centroids: return qid
    hint = clean(prompt_hint or "")
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio) if (hint and question_texts) else None
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted: return k
    return None

# ==============================
# SCORING
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts) -> pd.DataFrame:

    df_cols = list(df.columns)

    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id","staff_id")), None)
    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end","_submission_time","SubmissionDate","submissiondate"] if c in df.columns), None)

    dt_series = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series([pd.NaT]*len(df))
    start_dt  = pd.to_datetime(df[start_col], errors="coerce") if start_col else pd.Series([pd.NaT]*len(df))
    end_dt    = pd.to_datetime(df[end_col], errors="coerce")   if end_col   else pd.Series([pd.NaT]*len(df))
    duration_min = ((end_dt - start_dt).dt.total_seconds()/60.0).round(2)

    rows_out = []
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

    resolved_for_qid, _missing = {}, []
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit: resolved_for_qid[qid] = hit
        else:   _missing.append((qid, qhint))

    # Pre-embed distinct answers for speed
    distinct_answers = set()
    for _, resp in df.iterrows():
        for r in all_mapping:
            qid = r["question_id"]; col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(resp.get(col, ""))
                if a: distinct_answers.add(a)
    for t in distinct_answers: _ = embed_cached(t)

    for i, resp in df.iterrows():
        out = {}
        out["Date"] = (pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
                       if pd.notna(dt_series.iloc[i]) else str(i))
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr = {}
        any_ai_suspected = False
        ai_scores_this_row = []
        ai_flag_labels = []
        qtext_cache = {}

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            dfcol = resolved_for_qid.get(qid)
            if not dfcol or dfcol not in df.columns: continue

            ans = clean(resp.get(dfcol, ""))
            if not ans: continue
            vec = embed_cached(ans)

            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)
            if qkey and qkey not in qtext_cache:
                qtext_cache[qkey] = (by_qkey.get(qkey, {}) or {}).get("question_text","")
            qtext_for_ai = qtext_cache.get(qkey, "") if qkey else qhint

            sc = None
            if vec is not None:
                if qkey and qkey in q_centroids:
                    sims_q = {s: cos_sim(vec, v) for s, v in q_centroids[qkey].items() if v is not None}
                else:
                    sims_q = {}
                sims_a = {s: cos_sim(vec, v) for s, v in (attr_centroids.get(attr) or {}).items() if v is not None}
                sims_g = {s: cos_sim(vec, v) for s, v in global_centroids.items() if v is not None}

                def _pick_best(d): return max(d, key=d.get) if d else None
                sc = _pick_best(sims_q) or _pick_best(sims_a) or _pick_best(sims_g)

                if sc is not None:
                    base_q = qtext_cache.get(qkey, "")
                    if qa_overlap(ans, base_q or qhint) < MIN_QA_OVERLAP:
                        sc = min(sc, 1)

            # per-answer AI signal
            ai_score, flags = ai_signal_score(ans, qtext_for_ai)
            ai_scores_this_row.append(ai_score)
            if flags: ai_flag_labels.extend(flags)
            if ai_score >= AI_SUSPECT_THRESHOLD:
                any_ai_suspected = True

            # write score
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4): continue

            sk = f"{attr}_Qn{qn}"
            rk = f"{attr}_Rubric_Qn{qn}"
            if sc is None:
                out.setdefault(sk, ""); out.setdefault(rk, "")
            else:
                out[sk] = int(sc); out[rk] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

        # ensure Q placeholders exist
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute avgs + overall
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                out[f"{attr}_Avg (0–3)"] = ""
                out[f"{attr}_RANK"]      = ""
            else:
                avg = float(np.mean(scores)); band = int(round(avg))
                overall_total += band
                out[f"{attr}_Avg (0–3)"] = round(avg, 2)
                out[f"{attr}_RANK"]      = BANDS[band]

        out["Overall Total (0–24)"] = overall_total
        out["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        out["AI_suspected"] = bool(any_ai_suspected)
        out["AI_Score_Max"]  = round(max(ai_scores_this_row or [0.0]), 3)
        out["AI_Score_Mean"] = round(float(np.mean(ai_scores_this_row)) if ai_scores_this_row else 0.0, 3)
        out["AI_Flags_Top"]  = ", ".join(pd.Series(ai_flag_labels).value_counts().index[:3]) if ai_flag_labels else ""
        rows_out.append(out)

    res_df = pd.DataFrame(rows_out)

    def order_cols(cols):
        ordered = ["Date","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI_Score_Max", "AI_Score_Mean", "AI_Flags_Top", "AI_suspected"]
        extras = [c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols] + extras

    return res_df.reindex(columns=order_cols(list(res_df.columns)))

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
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False)
    return bio.getvalue()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME1", "Thought Leadership")

def _normalize_sa_dict(raw: dict) -> dict:
    if not raw: raise ValueError("gcp_service_account missing in secrets.")
    sa = dict(raw)
    if "token_ur" in sa and "token_uri" not in sa: sa["token_uri"] = sa.pop("token_ur")
    if sa.get("private_key") and "\\n" in sa["private_key"]: sa["private_key"] = sa["private_key"].replace("\\n", "\n")
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
    try:
        sh = gc.open_by_key(key)
    except gspread.SpreadsheetNotFound:
        raise ValueError(f"Spreadsheet with key '{key}' not found or not shared with the service account.")
    try:
        return sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        st.warning(f"Worksheet '{ws_name}' not found. Creating it…")
        return sh.add_worksheet(title=ws_name, rows="20000", cols="200")

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
                    "dimensions": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": cols}
                }
            }]
        })
    except Exception: pass

def upload_df_to_gsheets(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        df_out = _ensure_ai_last(df, export_name="AI_Suspected", source_name="AI_suspected")
        header = df_out.columns.astype(str).tolist()
        n_cols = len(header)
        if n_cols == 0: return False, "❌ No columns to write."
        values = df_out.astype(object).where(pd.notna(df_out), "").values.tolist()

        # start-only A1 ranges (avoid width mismatch)
        ws.clear()
        all_rows = [header] + values
        data_payload, start_row = [], 1
        for chunk in _chunk(all_rows, 10_000):
            a1_start = f"'{ws.title}'!A{start_row}"
            data_payload.append({"range": a1_start, "values": chunk})
            start_row += len(chunk)

        ws.spreadsheet.values_batch_update(body={"valueInputOption":"USER_ENTERED","data":data_payload})
        _post_write_formatting(ws, n_cols)
        return True, f"✅ Wrote {len(values)} rows × {n_cols} cols to '{ws.title}' (last='AI_Suspected')."
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"

# ==============================
# UI / MAIN (auto-run & auto-push)
# ==============================
def main():
    st.set_page_config(page_title="Thought Leadership Scoring", layout="wide")
    st.title("📊 Thought Leadership — Auto Scoring & Sheets Push")

    # 1) Load resources
    try:
        mapping = load_mapping_from_path(MAPPING_PATH)
    except Exception as e:
        st.error(f"Mapping load failed: {e}")
        return

    try:
        exemplars = read_jsonl_path(EXEMPLARS_PATH)
        if not exemplars: st.error(f"Exemplars empty: {EXEMPLARS_PATH}"); return
    except Exception as e:
        st.error(f"Exemplars load failed: {e}")
        return

    with st.spinner("Building semantic centroids..."):
        q_c, a_c, g_c, by_q, qtexts = build_centroids(exemplars)

    # 2) Fetch data
    with st.spinner("Fetching Kobo submissions (or local export)..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No data to score."); return

    st.caption("Sample of raw data:")
    st.dataframe(df.head(10), use_container_width=True)

    # 3) Score
    with st.spinner("Scoring responses (+ AI detection)..."):
        scored_df = score_dataframe(df, mapping, q_c, a_c, g_c, by_q, qtexts)

    st.success("✅ Scoring complete.")
    st.dataframe(scored_df.head(50), use_container_width=True)

    # 4) Auto push to Sheets
    with st.spinner("📤 Pushing to Google Sheets..."):
        ok, msg = upload_df_to_gsheets(scored_df)
    show_status(ok, msg)

    # 5) Offer download (no button needed to run; just download link)
    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(scored_df),
        file_name="Leadership_Scoring.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
