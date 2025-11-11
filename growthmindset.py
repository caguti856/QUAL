# advisory.py — Growth Mindset Scoring (auto-run, full source columns, Date→Duration→Care_Staff first, AI last)

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
# SECRETS / PATHS
# ==============================
KOBO_BASE       = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID2  = st.secrets.get("KOBO_ASSET_ID2", "")
KOBO_TOKEN      = st.secrets.get("KOBO_TOKEN", "")
AUTO_PUSH       = bool(st.secrets.get("AUTO_PUSH", False))
AUTO_RUN        = True  # no buttons

DATASETS_DIR    = Path("DATASETS")
MAPPING_PATH    = DATASETS_DIR / "mapping2.csv"
EXEMPLARS_PATH  = DATASETS_DIR / "growthmindset.jsonl"

# ==============================
# CONSTANTS
# ==============================
BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 21, 24),
    ("Strategic Advisor",       16, 20),
    ("Emerging Advisor",        10, 15),
    ("Needs Capacity Support",   0,  9),
]

ORDERED_ATTRS = [
    "Learning Agility",
    "Digital Savvy",
    "Innovation",
    "Contextual Intelligence",
]

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

# passthrough source columns we keep (front pool; we’ll later filter out an explicit exclude set)
PASSTHROUGH_HINTS = [
    "staff id","staff_id","staffid","_id","id","_uuid","uuid","instanceid","_submission_time",
    "submissiondate","submission_date","start","_start","end","_end","today","date","deviceid",
    "username","enumerator","submitted_via_web","_xform_id_string","formid","assetid","care_staff"
]

# EXCLUDE these specific raw source cols from the visible table (your list)
_EXCLUDE_SOURCE_COLS_LOWER = {
    "_id","formhub/uuid","start","end","today","staff_id","meta/instanceid",
    "_xform_id_string","_uuid","meta/rootuuid","_submission_time","_validation_status"
}

# ==============================
# AI DETECTION (aggressive, same style)
# ==============================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

TRANSITION_OPEN_RX = re.compile(r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b", re.I)
LIST_CUES_RX       = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX          = re.compile(r"^[-*•]\s", re.M)
SYMBOL_RX          = re.compile(r"[—–\-]{2,}|[≥≤≧≦≈±×÷%]|[→←⇒↔↑↓]|[•●◆▶✓✔✗❌§†‡]", re.U)
TIMEBOX_RX         = re.compile(r"(?:\bday\s*\d+\b|\bweek\s*\d+\b|\bmonth\s*\d+\b|\bquarter\s*\d+\b|\b\d+\s*(?:days?|weeks?|months?|quarters?)\b|\bby\s+day\s*\d+\b)", re.I)
AI_RX              = re.compile(r"(?:as an ai\b|i am an ai\b)", re.I)
AI_BUZZWORDS = {
    "minimum viable","feedback loop","trade-off","evidence-based",
    "stakeholder alignment","learners’ agency","norm shifts","quick win",
    "low-lift","scalable","best practice","pilot theatre","timeboxed"
}

def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def qa_overlap(ans: str, qtext: str) -> float:
    at = set(re.findall(r"\w+", (ans or "").lower()))
    qt = set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

def _avg_sentence_len(text: str) -> float:
    sents = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    if not sents: return 0.0
    toks = re.findall(r"\w+", text or "")
    return len(toks) / max(len(sents), 1)

def _type_token_ratio(text: str) -> float:
    toks = [t.lower() for t in re.findall(r"[a-z]+", text or "")]
    return 1.0 if not toks else len(set(toks))/len(toks)

def ai_signal_score(text: str, question_hint: str = "") -> float:
    t = clean(text)
    if not t: return 0.0
    score = 0.0
    if SYMBOL_RX.search(t):   score += 0.35
    if TIMEBOX_RX.search(t):  score += 0.15
    if AI_RX.search(t):       score += 0.35
    if TRANSITION_OPEN_RX.search(t): score += 0.12
    if LIST_CUES_RX.search(t):       score += 0.12
    if BULLET_RX.search(t):          score += 0.08
    buzz_hits = sum(1 for b in AI_BUZZWORDS if b in t.lower())
    if buzz_hits: score += min(0.24, 0.08*buzz_hits)
    asl = _avg_sentence_len(t)
    if   asl >= 26: score += 0.18
    elif asl >= 18: score += 0.10
    ttr = _type_token_ratio(t)
    if ttr <= 0.45 and len(t) >= 180: score += 0.10
    if question_hint:
        overlap = qa_overlap(t, question_hint)
        if overlap < 0.06: score += 0.10
    return max(0.0, min(1.0, score))

# ==============================
# KOBO
# ==============================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID2 or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID2 and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions","data"):
        url = kobo_url(KOBO_ASSET_ID2, kind)
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 404: continue
            r.raise_for_status()
            payload = r.json()
            results = payload if isinstance(payload, list) else payload.get("results", [])
            if not results and "results" not in payload: results = payload
            df = pd.DataFrame(results)
            if not df.empty: df.columns = [str(c).strip() for c in df.columns]
            return df
        except requests.HTTPError:
            if r.status_code in (401,403):
                st.error("Kobo auth failed: check KOBO_TOKEN and tenant."); return pd.DataFrame()
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}"); return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}"); return pd.DataFrame()
    st.error("Could not fetch data. Check KOBO_BASE/ASSET/TOKEN.")
    return pd.DataFrame()

# ==============================
# MAPPING + EXEMPLARS (Growth Mindset)
# ==============================
QID_PREFIX_TO_SECTION = {"LA":"A1","DS":"A2","IN":"A3","CI":"A4"}
QNUM_RX = re.compile(r"_Q(\d+)$")

def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), "mapping needs: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    # keep only desired attrs
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists(): raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s: rows.append(json.loads(s))
    return rows

def build_kobo_bases_from_qid(question_id: str) -> list[str]:
    if not question_id: return []
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m: return []
    qn = m.group(1)
    prefix = qid.split("_Q")[0]
    sect = QID_PREFIX_TO_SECTION.get(prefix)
    if not sect: return []
    roots = ["Growthmindset","Growth Mindset"]
    return [f"{root}/{sect}_Section/{sect}_{qn}" for root in roots]

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
    if "growthmindset/" in c or "growth mindset/" in c or "/a" in c: s += 2
    return s

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    cols_original = list(df_cols)
    cols_lower = {c.lower(): c for c in cols_original}

    bases = build_kobo_bases_from_qid(question_id)
    for base in bases:
        if base.lower() in cols_lower: return cols_lower[base.lower()]
        for v in expand_possible_kobo_columns(base):
            if v.lower() in cols_lower: return cols_lower[v.lower()]
        for c in cols_original:
            if c.lower().startswith(base.lower()): return c

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
        for c in cols_original:
            sc = _score_kobo_header(c, token)
            if sc > bs: bs, best = sc, c
        if best and bs >= 82: return best

    hint = clean(prompt_hint or "")
    if hint:
        cands = [(c, c.lower()) for c in cols_original]
        hits = process.extract(hint.lower(), [lo for _, lo in cands],
                               scorer=fuzz.partial_token_set_ratio, limit=5)
        for _, lo, score in hits:
            if score >= 88:
                for orig, low in cands:
                    if low == lo: return orig
    return None

# ==============================
# EMBEDDINGS (batched, cached)
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

def resolve_qkey(q_centroids, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in q_centroids: return qid
    hint = clean(prompt_hint or "")
    if not (hint and question_texts): return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted: return k
    return None

# ==============================
# SCORING (per-question + averages; dynamic Qn)
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
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

    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end","_submission_time","SubmissionDate","submissiondate"] if c in df.columns), None)

    dt_series = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series([pd.NaT]*len(df))
    start_dt  = pd.to_datetime(df[start_col], errors="coerce") if start_col else pd.Series([pd.NaT]*len(df))
    end_dt    = pd.to_datetime(df[end_col], errors="coerce")   if end_col   else pd.Series([pd.NaT]*len(df))
    duration_min = ((end_dt - start_dt).dt.total_seconds()/60.0).round(2)

    # mapping resolution
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]
    resolved_for_qid = {}
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit: resolved_for_qid[qid] = hit

    # collect all question numbers present per attribute (dynamic)
    QNUM_RX_LOCAL = re.compile(r"_Q(\d+)$", re.I)
    attr_to_qnums: dict[str, set[int]] = {a: set() for a in ORDERED_ATTRS}
    for r in all_mapping:
        qid = (r.get("question_id") or "").strip()
        attr = r["attribute"]
        m = QNUM_RX_LOCAL.search(qid)
        if m:
            try: attr_to_qnums[attr].add(int(m.group(1)))
            except: pass

    # batch-embed all distinct answers once
    distinct_answers = set()
    for _, row in df.iterrows():
        for r in all_mapping:
            col = resolved_for_qid.get(r["question_id"])
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a: distinct_answers.add(a)
    embed_many(list(distinct_answers))

    def _best_sim(vec, cent_dict):
        best_s, best_v = None, -1e9
        for s, c in cent_dict.items():
            if c is None: continue
            v = float(np.dot(vec, c))
            if v > best_v: best_v, best_s = v, s
        return best_s

    out_rows = []
    for i, resp in df.iterrows():
        row = {}

        # Date, Duration, Care_Staff
        row["Date"] = (pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
                       if pd.notna(dt_series.iloc[i]) else str(i))
        row["Duration"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""
        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(resp.get(who_col)) if who_col else ""

        # passthrough original source columns (minus excluded set and our first three)
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER: continue
            if c in ("Date","Duration","Care_Staff"): continue
            row[c] = resp.get(c, "")

        per_attr = {a: [] for a in ORDERED_ATTRS}
        any_ai = False
        qtext_cache = {}

        # row answers cache
        row_answers = {}
        for r in all_mapping:
            qid = r["question_id"]; col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                row_answers[qid] = clean(resp.get(col, ""))

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            col = resolved_for_qid.get(qid)
            if not col: continue
            ans = row_answers.get(qid, "")
            if not ans: continue
            vec = emb_of(ans)

            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)
            if qkey and qkey not in qtext_cache:
                qtext_cache[qkey] = (by_qkey.get(qkey, {}) or {}).get("question_text","")
            qhint_full = qtext_cache.get(qkey, "") if qkey else qhint

            sc = None
            if vec is not None:
                if qkey and qkey in q_centroids:
                    sc = _best_sim(vec, q_centroids[qkey])
                if sc is None and attr in attr_centroids:
                    sc = _best_sim(vec, attr_centroids[attr])
                if sc is None:
                    sc = _best_sim(vec, global_centroids)
                if sc is not None and qa_overlap(ans, qhint_full or qhint) < MIN_QA_OVERLAP:
                    sc = min(sc, 1)

            # AI suspicion
            ai_score = ai_signal_score(ans, qhint_full)
            if ai_score >= AI_SUSPECT_THRESHOLD:
                any_ai = True

            # write per-question score & rubric (for ANY Qn found)
            m = QNUM_RX_LOCAL.search(qid or "")
            qn = int(m.group(1)) if m else None
            if sc is not None and qn is not None:
                sk = f"{attr}_Qn{qn}"
                rk = f"{attr}_Rubric_Qn{qn}"
                row[sk] = int(sc)
                row[rk] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

        # fill missing Qn/rubric fields to keep stable shape (all qnums from mapping)
        for attr in ORDERED_ATTRS:
            for qn in sorted(attr_to_qnums[attr]):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + ranks (over all scored items)
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                row[f"{attr}_Avg (0–3)"] = ""
                row[f"{attr}_RANK"]      = ""
            else:
                avg = float(np.mean(scores)); band = int(round(avg))
                overall_total += band
                row[f"{attr}_Avg (0–3)"] = round(avg, 2)
                row[f"{attr}_RANK"]      = BANDS[band]

        row["Overall Total (0–24)"] = overall_total
        row["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        row["AI_suspected"] = bool(any_ai)
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # ---- final column order (dynamic) ----
    # 1) Date, Duration, Care_Staff
    ordered = [c for c in ["Date","Duration","Care_Staff"] if c in res.columns]

    # 2) All original source columns (original order), minus excluded ones and minus our first three
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date","Duration","Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    # 3) Per-question blocks (dynamic qn sets)
    mid_q = []
    for attr in ORDERED_ATTRS:
        for qn in sorted(attr_to_qnums[attr]):
            mid_q += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
    ordered += [c for c in mid_q if c in res.columns]

    # 4) Attribute avgs + ranks
    mid_a = []
    for attr in ORDERED_ATTRS:
        mid_a += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += [c for c in mid_a if c in res.columns]

    # 5) Overall then AI (AI must be right after Overall Rank AND be the last column)
    ordered += [c for c in ["Overall Total (0–24)","Overall Rank"] if c in res.columns]
    if "AI_suspected" in res.columns:
        ordered += ["AI_suspected"]

    # enforce ordering (extras hidden to keep AI last)
    res = res.reindex(columns=ordered)

    return res

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
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df_out.to_excel(w, index=False)
    return bio.getvalue()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME2", "Growthmindset")

def _normalize_sa_dict(raw: dict) -> dict:
    if not raw: raise ValueError("gcp_service_account missing in secrets.")
    sa = dict(raw)
    if "token_ur" in sa and "token_uri" not in sa: sa["token_uri"] = sa.pop("token_ur")
    if sa.get("private_key") and "\\n" in sa["private_key"]:
        sa["private_key"] = sa["private_key"].replace("\\n","\n")
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
    sh = gc.open_by_key(key)
    try:
        return sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=ws_name, rows="20000", cols="150")

def _post_write_formatting(ws: gspread.Worksheet, cols: int) -> None:
    try: ws.freeze(rows=1)
    except Exception: pass
    try:
        ws.spreadsheet.batch_update({
            "requests":[{"autoResizeDimensions":{
                "dimensions":{"sheetId": ws.id, "dimension":"COLUMNS","startIndex":0,"endIndex":cols}
            }}]
        })
    except Exception: pass

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
    st.title("📊 Growth Mindset Scoring — Auto (Full Source + Per-Question)")

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

    with st.spinner("Building semantic centroids..."):
        q_c, a_c, g_c, by_q, qtexts = build_centroids(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found.")
        return

    # show full fetched dataset
    st.subheader("Fetched dataset (all rows)")
    st.caption(f"Rows: {len(df):,}  •  Columns: {len(df.columns):,}")
    st.dataframe(df, use_container_width=True)

    with st.spinner("Scoring (+ AI detection)..."):
        scored = score_dataframe(df, mapping, q_c, a_c, g_c, by_q, qtexts)

    st.success("✅ Scoring complete.")

    st.subheader("Scored table (all rows)")
    st.caption("Date → Duration → Care_Staff, then original source columns (minus excluded), dynamic per-question scores & rubrics, attribute averages, Overall, Overall Rank, and AI_suspected last.")
    st.dataframe(scored, use_container_width=True)

    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(scored),
        file_name="Growthmindset_Scoring.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    st.download_button(
        "⬇️ Download CSV",
        data=_ensure_ai_last(scored).to_csv(index=False).encode("utf-8"),
        file_name="Growthmindset_Scoring.csv",
        mime="text/csv",
        use_container_width=True
    )

    if AUTO_PUSH:
        with st.spinner("📤 Sending to Google Sheets..."):
            ok, msg = upload_df_to_gsheets(scored)
        (st.success if ok else st.error)(msg)

if __name__ == "__main__":
    main()
