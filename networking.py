import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

import json, re, unicodedata
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from io import BytesIO

# ==============================
# CONSTANTS / PATHS
# ==============================
KOBO_BASE         = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID3    = st.secrets.get("KOBO_ASSET_ID3", "")
KOBO_TOKEN        = st.secrets.get("KOBO_TOKEN", "")
AUTO_PUSH         = bool(st.secrets.get("AUTO_PUSH", False))
AUTO_RUN          = bool(st.secrets.get("AUTO_RUN", True))  # default True

DATASETS_DIR      = Path("DATASETS")
MAPPING_PATH      = DATASETS_DIR / "mapping3.csv"
EXEMPLARS_PATH    = DATASETS_DIR / "networking.jsonl"

BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Localization Champion", 21, 24),
    ("Skilled Networked Advocate", 16, 20),
    ("Developing Influencer", 10, 15),
    ("Needs Support", 0, 9),
]

ORDERED_ATTRS = [
    "Strategic Positioning & Donor Fluency",
    "Power-Aware Stakeholder Mapping",
    "Equitable Allyship & Local Fronting",
    "Coalition Governance & Convening",
    "Community-Centered Messaging",
    "Evidence-Led Learning (Local Knowledge)",
    "Influence Without Authority",
    "Risk Management & Adaptive Communication",
]

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

# passthrough source columns we keep (front pool; we later filter an explicit exclude set)
PASSTHROUGH_HINTS = [
    "staff id","staff_id","staffid","_id","id","_uuid","uuid","instanceid","_submission_time",
    "submissiondate","submission_date","start","_start","end","_end","today","date","deviceid",
    "username","enumerator","submitted_via_web","_xform_id_string","formid","assetid","care_staff"
]

# EXCLUDE these specific raw source cols from the visible table
_EXCLUDE_SOURCE_COLS_LOWER = {
    "_id","formhub/uuid","start","end","today","staff_id","meta/instanceid",
    "_xform_id_string","_uuid","meta/rootuuid","_submission_time","_validation_status"
}

# ==============================
# AI DETECTION (robust, lightweight)
# ==============================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

TRANSITION_OPEN_RX = re.compile(r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b", re.I)
LIST_CUES_RX       = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX          = re.compile(r"^[-*•]\s", re.M)
SYMBOL_RX          = re.compile(r"[—–\-]{2,}|[≥≤≧≦≈±×÷%]|[→←⇒↔↑↓]|[•●◆▶✓✔✗❌§†‡]", re.U)
TIMEBOX_RX         = re.compile(r"(?:\bday\s*\d+\b|\bweek\s*\d+\b|\bmonth\s*\d+\b|\bquarter\s*\d+\b|\b\d+\s*(?:days?|weeks?|months?|quarters?)\b|\bby\s+day\s*\d+\b)", re.I)
AI_META_RX         = re.compile(r"(?:as an ai\b|i am an ai\b)", re.I)
AI_BUZZWORDS = {
    "minimum viable","feedback loop","trade-off","evidence-based",
    "stakeholder alignment","learners’ agency","norm shifts","quick win",
    "low-lift","scalable","best practice","pilot theatre","timeboxed"
}

def _avg_sentence_len(text: str) -> float:
    sents = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    if not sents: return 0.0
    toks = re.findall(r"\w+", text or "")
    return len(toks) / max(len(sents), 1)

def _type_token_ratio(text: str) -> float:
    toks = [t.lower() for t in re.findall(r"[a-z]+", text or "")]
    return 1.0 if not toks else len(set(toks))/len(toks)

def ai_signal_score(text: str, question_hint: str = "") -> float:
    t = (text or "").strip()
    if not t: return 0.0
    score = 0.0
    if SYMBOL_RX.search(t):   score += 0.35
    if TIMEBOX_RX.search(t):  score += 0.15
    if AI_META_RX.search(t):  score += 0.35
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
        at = set(re.findall(r"\w+", t.lower()))
        qt = set(re.findall(r"\w+", question_hint.lower()))
        overlap = (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0
        if overlap < 0.06: score += 0.10
    return max(0.0, min(1.0, score))

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

def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

def normalize_col_name(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("’","'").replace("“","\"").replace("”","\"")
    s = re.sub(r"\s+"," ", s)
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    return s

def show_status(ok: bool, msg: str) -> None:
    if ok: st.success(msg)
    else:  st.error(msg)

# ==============================
# LOADERS
# ==============================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), \
        "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID3 or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID3 and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions","data"):
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
        except requests.HTTPError:
            if r.status_code in (401, 403):
                st.error("Kobo auth failed: check KOBO_TOKEN and tenant.")
                return pd.DataFrame()
            if r.status_code == 404:
                continue
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()
    st.error("Could not fetch data. Check KOBO_BASE, KOBO_ASSET_ID3, token permissions.")
    return pd.DataFrame()

# ==============================
# QUESTION_ID → KOBO COLUMN RESOLVER
# ==============================
QID_PREFIX_TO_SECTION = {"SPD":"A1","PAS":"A2","EAL":"A3","CGC":"A4","CCM":"A5","ELL":"A6","IWA":"A7","RMA":"A8"}
QNUM_RX = re.compile(r"_Q(\d+)$")

def build_kobo_bases_from_qid(question_id: str) -> list[str]:
    if not question_id:
        return []
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m:
        return []
    qn = m.group(1)
    prefix = qid.split("_Q")[0]
    if prefix not in QID_PREFIX_TO_SECTION:
        return []
    section = QID_PREFIX_TO_SECTION[prefix]
    roots = ["networking","Networking"]
    return [f"{root}/{section}_Section/{section}_{qn}" for root in roots]

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
    c = col.lower()
    t = token.lower()
    if c == t: return 100
    score = 0
    if c.endswith("/" + t): score = max(score, 95)
    if f"/{t}/" in c:       score = max(score, 92)
    if (f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c): score = max(score, 90)
    if t in c:              score = max(score, 80)
    if "english" in c or "label" in c or "(en)" in c: score += 5
    if "answer (text)" in c or "answer_text" in c or "text" in c: score += 5
    if "growthmindset/" in c or "networking/" in c or "/a" in c: score += 2
    return score

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    cols_original = list(df_cols)
    cols_lower_map = {c.lower(): c for c in cols_original}

    bases = build_kobo_bases_from_qid(question_id)
    for base in bases:
        if base.lower() in cols_lower_map:
            return cols_lower_map[base.lower()]
        for v in expand_possible_kobo_columns(base):
            if v.lower() in cols_lower_map:
                return cols_lower_map[v.lower()]
        for c in cols_original:
            if c.lower().startswith(base.lower()):
                return c

    token = None
    if question_id:
        qid = question_id.strip().upper()
        m = QNUM_RX.search(qid)
        if m:
            qn = m.group(1)
            prefix = qid.split("_Q")[0]
            sect = QID_PREFIX_TO_SECTION.get(prefix)
            if sect:
                token = f"{sect}_{qn}"

    if token:
        best_col, best_score = None, 0
        for col in cols_original:
            s = _score_kobo_header(col, token)
            if s > best_score:
                best_score, best_col = s, col
        if best_col and best_score >= 82:
            return best_col

    hint = clean(prompt_hint or "")
    if hint:
        candidates = [(c, c.lower()) for c in cols_original]
        hits = process.extract(hint.lower(), [lo for _, lo in candidates],
                               scorer=fuzz.partial_token_set_ratio, limit=5)
        for _, lo, score in hits:
            if score >= 88:
                for orig, low in candidates:
                    if low == lo:
                        return orig
    return None

# ==============================
# EMBEDDINGS (batched + cached for speed)
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
    missing = [clean(t) for t in texts if t and clean(t) not in _EMB_CACHE]
    if not missing: return
    pack = _embed_texts_cached(tuple(missing))
    _EMB_CACHE.update(pack)

def emb_of(text: str):
    return _EMB_CACHE.get(clean(text or ""), None)

def build_centroids(exemplars: list[dict]):
    by_qkey, by_attr, question_texts = {}, {}, []
    all_texts = []
    for e in exemplars:
        qid   = clean(e.get("question_id",""))
        qtext = clean(e.get("question_text",""))
        score = int(e.get("score",0))
        text  = clean(e.get("text",""))
        attr  = clean(e.get("attribute",""))
        if not qid and not qtext:
            continue
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
    if qid and qid in q_centroids:
        return qid
    hint = clean(prompt_hint or "")
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio) if (hint and question_texts) else None
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted:
                return k
    return None

# ==============================
# SCORING
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts):

    df_cols = list(df.columns)

    with st.expander("🔎 Debug: Networking section columns present", expanded=False):
        sample_cols = [c for c in df_cols if "/A" in c or "Networking/" in c or c.startswith("A")]
        st.write(sample_cols[:80])

    # identify useful passthrough cols (original order)
    def want_col(c):
        lc = c.strip().lower()
        return any(h in lc for h in PASSTHROUGH_HINTS)
    passthrough_cols = []
    seen = set()
    for c in df_cols:
        if want_col(c) and c not in seen:
            passthrough_cols.append(c); seen.add(c)

    # preferred staff id column(s)
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

    # resolve mapping to Kobo columns
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]
    resolved_for_qid = {}
    missing_map_rows = []
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit:
            resolved_for_qid[qid] = hit
        else:
            missing_map_rows.append((qid, qhint))
    with st.expander("🧭 Mapping → Kobo column resolution (by question_id)", expanded=False):
        if resolved_for_qid:
            show = list(resolved_for_qid.items())[:80]
            st.dataframe(pd.DataFrame(show, columns=["question_id","kobo_column"]))
        if missing_map_rows:
            st.warning(f"{len(missing_map_rows)} question_ids not found in Kobo headers (showing up to 30).")
            st.dataframe(pd.DataFrame(missing_map_rows[:30], columns=["question_id","prompt_hint"]))

    # batch-embed all distinct answers ONCE (major speed-up)
    distinct_answers = set()
    for _, resp in df.iterrows():
        for r in all_mapping:
            qid = r["question_id"]
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(resp.get(col, ""))
                if a: distinct_answers.add(a)
    embed_many(list(distinct_answers))

    out_rows = []
    for i, resp in df.iterrows():
        out = {}

        # ----- front fields -----
        if pd.notna(dt_series.iloc[i]):
            out["Date"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["Date"] = str(i)

        who_col = care_staff_col or staff_id_col
        out["Staff ID"] = str(resp.get(who_col)) if who_col else ""
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        # passthrough (excluding some system fields and our front 3)
        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date","Duration_min","Staff ID"):
                continue
            out[c] = resp.get(c, "")

        # ----- per-question scoring -----
        per_attr = {}
        ai_scores_row = []
        qtext_cache = {}

        # per-row answer cache
        row_answers = {}
        for r in all_mapping:
            qid = r["question_id"]
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                row_answers[qid] = clean(resp.get(col, ""))

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            col = resolved_for_qid.get(qid)
            if not col:
                continue
            ans = row_answers.get(qid, "")
            if not ans:
                continue

            vec = emb_of(ans)

            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)
            if qkey and qkey not in qtext_cache:
                qtext_cache[qkey] = (by_qkey.get(qkey, {}) or {}).get("question_text","")
            qhint_full = qtext_cache.get(qkey, "") if qkey else qhint

            sc = None
            if vec is not None:
                def best_sim(cent_dict):
                    best_s, best_v = None, -1e9
                    for s, c in cent_dict.items():
                        if c is None: continue
                        v = float(np.dot(vec, c))
                        if v > best_v: best_v, best_s = v, s
                    return best_s
                if qkey and qkey in q_centroids:
                    sc = best_sim(q_centroids[qkey])
                if sc is None and attr in attr_centroids:
                    sc = best_sim(attr_centroids[attr])
                if sc is None:
                    sc = best_sim(global_centroids)
                if sc is not None and qa_overlap(ans, qhint_full or qhint) < MIN_QA_OVERLAP:
                    sc = min(sc, 1)

            # AI suspicion (per-answer), accumulate
            ai_score = ai_signal_score(ans, qhint_full)
            ai_scores_row.append(ai_score)

            # write per-question (Q1..Q4 only)
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn in (1,2,3,4) and sc is not None:
                sk = f"{attr}_Qn{qn}"
                rk = f"{attr}_Rubric_Qn{qn}"
                out[sk] = int(sc)
                out[rk] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

        # ensure stable shape
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + ranks
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

        # AI-Suspected as boolean; LAST column later in orderer
        out["AI-Suspected"] = bool(any(s >= AI_SUSPECT_THRESHOLD for s in ai_scores_row))

        out_rows.append(out)

    res_df = pd.DataFrame(out_rows)

    # ----- final column order -----
    def order_cols(cols):
        ordered = ["Date","Staff ID","Duration_min"]  # Date first
        # keep passthrough (except excluded and front three)
        source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
        source_cols = [c for c in source_cols if c not in ("Date","Duration_min","Staff ID")]
        ordered += [c for c in source_cols if c in cols]
        # question blocks
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
        # attribute avgs + ranks
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        # overall + AI last
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI-Suspected"]
        extras = [c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols] + extras

    return res_df.reindex(columns=order_cols(list(res_df.columns)))

# ==============================
# EXPORTS
# ==============================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return bio.getvalue()

# ==============================
# Google Sheets (clean)
# ==============================
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
    try:
        sh = gc.open_by_key(key)
    except gspread.SpreadsheetNotFound:
        raise ValueError(f"Spreadsheet with key '{key}' not found or not shared with the service account.")
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
        col_end = _to_a1_col(cols)
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
def _run_pipeline():
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

    with st.spinner("Building semantic centroids..."):
        q_centroids, attr_centroids, global_centroids, by_qkey, question_texts = build_centroids(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()

    if df.empty:
        st.warning("No Kobo submissions found.")
        return

    st.caption("Fetched sample:")
    st.dataframe(df.head(), use_container_width=True)

    with st.spinner("Scoring responses..."):
        scored_df = score_dataframe(df, mapping, q_centroids, attr_centroids, global_centroids, by_qkey, question_texts)

    st.success("✅ Scoring complete.")
    st.dataframe(scored_df.head(50), use_container_width=True)
    st.session_state["scored_df"] = scored_df

    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(scored_df),
        file_name="Networking_Scoring.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    st.download_button(
        "⬇️ Download CSV",
        data=scored_df.to_csv(index=False).encode("utf-8"),
        file_name="Networking_Scoring.csv",
        mime="text/csv",
        use_container_width=True
    )

    if AUTO_PUSH:
        with st.spinner("📤 Sending to Google Sheets..."):
            ok, msg = upload_df_to_gsheets(scored_df)
        (st.success if ok else st.error)(msg)

def main():
    st.title("📊 Networking & Advocacy — Kobo → Scored (AI-aware) → Excel/Sheets")
    if AUTO_RUN:
        _run_pipeline()
    else:
        if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True):
            _run_pipeline()

if __name__ == "__main__":
    main()
