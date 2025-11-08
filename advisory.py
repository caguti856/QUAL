# advisory.py — one-file pipeline: Kobo -> Score (+confidence/router) -> Sheets
# Streamlit page exposing:
#   - "🚀 Fetch Kobo & Score" button
#   - Download Excel
#   - Push full + autos_accepted + review_queue (+ optional star schema) to Google Sheets

import streamlit as st
import json, re, unicodedata
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests

import gspread
from google.oauth2.service_account import Credentials

from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# ==============================
# CONSTANTS / PATHS / SECRETS
# ==============================
KOBO_BASE        = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID    = st.secrets.get("KOBO_ASSET_ID", "")
KOBO_TOKEN       = st.secrets.get("KOBO_TOKEN", "")

DATASETS_DIR     = Path("DATASETS")
MAPPING_PATH     = DATASETS_DIR / "mapping.csv"
EXEMPLARS_PATH   = DATASETS_DIR / "advisory_exemplars_smart.cleaned.jsonl"

# Scoring bands
BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 21, 24),
    ("Strategic Advisor",       16, 20),
    ("Emerging Advisor",        10, 15),
    ("Needs Capacity Support",   0,  9),
]

# Attribute order (used for column ordering & averages)
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

# Matching & QA heuristics
FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

# 95/5 router tuning
MIN_CONF_AUTO = 0.78     # raise for stricter auto; lower for more auto
MAX_DISAGREE  = 1        # if global vs chosen band differs by >1 => review

# Google Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME", "Advisory")

# ==============================
# SMALL HELPERS
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

AI_RX = re.compile(r"(?:-{3,}|—{2,}|_{2,}|\.{4,}|as an ai\b|i am an ai\b)", re.I)
def looks_ai_like(t): return bool(AI_RX.search(clean(t)))

def show_status(ok: bool, msg: str) -> None:
    if ok: st.success(msg)
    else:  st.error(msg)

# ==============================
# LOADERS (mapping / exemplars)
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

# ==============================
# KOBO FETCH
# ==============================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions","data"):
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
    st.error("Could not fetch data. Check KOBO_BASE, KOBO_ASSET_ID, token permissions.")
    return pd.DataFrame()

# ==============================
# QID → KOBO COLUMN RESOLVER
# ==============================
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
QNUM_RX = re.compile(r"_Q(\d+)$")

def build_kobo_base_from_qid(question_id: str) -> str | None:
    if not question_id:
        return None
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m:
        return None
    qn = m.group(1)
    prefix = qid.split("_Q")[0]
    if prefix not in QID_PREFIX_TO_SECTION:
        return None
    section = QID_PREFIX_TO_SECTION[prefix]
    return f"Advisory/{section}_Section/{section}_{qn}"

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
    if c == t:
        return 100
    score = 0
    if c.endswith("/" + t): score = max(score, 95)
    if f"/{t}/" in c:       score = max(score, 92)
    if f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c: score = max(score, 90)
    if t in c:              score = max(score, 80)
    if "english" in c or "label" in c: score += 3
    if "answer (text)" in c or "answer_text" in c or "text" in c: score += 2
    if "advisory/" in c or "/a" in c: score += 1
    return score

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    base = build_kobo_base_from_qid(question_id)
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
    if base and base in df_cols:
        return base
    if base:
        for v in expand_possible_kobo_columns(base):
            if v in df_cols: return v
        for c in df_cols:
            if c.startswith(base): return c
    if token:
        best_col, best_score = None, 0
        for col in df_cols:
            s = _score_kobo_header(col, token)
            if s > best_score:
                best_score, best_col = s, col
        if best_col and best_score >= 82:
            return best_col
    hint = clean(prompt_hint or "")
    if hint:
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=5)
        for col, score, _ in hits:
            if score >= 88:
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

    embedder = get_embedder()

    def centroid(texts):
        if not texts: return None
        embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        # L2-normalized; mean keeps them roughly in unit ball
        return embs.mean(axis=0)

    def build_centroids_for_q(texts, scores):
        buckets = {0:[],1:[],2:[],3:[]}
        for t,s in zip(texts, scores):
            if t: buckets[int(s)].append(t)
        return {sc: centroid(batch) for sc, batch in buckets.items()}

    q_centroids = {k: build_centroids_for_q(v["texts"], v["scores"]) for k, v in by_qkey.items()}
    attr_centroids = {attr: {sc: centroid(txts) for sc, txts in bucks.items()} for attr, bucks in by_attr.items()}

    global_buckets = {0:[],1:[],2:[],3:[]}
    for e in exemplars:
        sc = int(e.get("score",0)); txt = clean(e.get("text",""))
        if txt: global_buckets[sc].append(txt)
    global_centroids = {sc: centroid(txts) for sc, txts in global_buckets.items()}

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
# CONFIDENCE / ROUTER (95/5)
# ==============================
def _softmax_like(scores: dict[int, float]) -> dict[int, float]:
    import math
    if not scores: return {}
    vals = list(scores.values())
    m = max(vals)
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    s = sum(exps.values())
    return {k: (v/s) for k, v in exps.items()} if s > 0 else {k:0.0 for k in scores}

def _pick_with_conf(q_sims: dict[int,float], a_sims: dict[int,float], g_sims: dict[int,float]):
    # weight by source trust
    qw = 1.0 if q_sims else 0.0
    aw = 0.6 if a_sims else 0.0
    gw = 0.4 if g_sims else 0.0
    qp = _softmax_like(q_sims) if q_sims else {}
    ap = _softmax_like(a_sims) if a_sims else {}
    gp = _softmax_like(g_sims) if g_sims else {}
    mix = {b: qw*qp.get(b,0)+aw*ap.get(b,0)+gw*gp.get(b,0) for b in (0,1,2,3)}
    band = max(mix, key=mix.get)
    conf = float(mix[band])
    # provenance for debug
    src_used = "global"
    if q_sims and max(q_sims.values()) >= max(a_sims.values() if a_sims else [-9e9]) and max(q_sims.values()) >= max(g_sims.values() if g_sims else [-9e9]):
        src_used = "question"
    elif a_sims and max(a_sims.values()) >= max(g_sims.values() if g_sims else [-9e9]):
        src_used = "attribute"
    return band, conf, src_used

def _route(band: int, conf: float, band_global: int | None, risky: bool) -> tuple[str, str]:
    if risky:
        return "review", "AI-like pattern"
    if band_global is not None and abs(band - band_global) > MAX_DISAGREE:
        return "review", "Disagreement with global centroid"
    if conf >= MIN_CONF_AUTO:
        return "auto", "High confidence"
    return "review", "Low confidence"

# ==============================
# SCORER
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts):

    df_cols = list(df.columns)

    with st.expander("🔎 Debug: Advisory section columns present", expanded=False):
        sample_cols = [c for c in df_cols if "/A" in c or "Advisory/" in c or c.startswith("A")]
        st.write(sample_cols[:80])

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
            show = list(resolved_for_qid.items())[:60]
            st.dataframe(pd.DataFrame(show, columns=["question_id","kobo_column"]))
        if missing_map_rows:
            st.warning(f"{len(missing_map_rows)} question_ids not found in Kobo headers (showing up to 30).")
            st.dataframe(pd.DataFrame(missing_map_rows[:30], columns=["question_id","prompt_hint"]))

    # pre-embed all distinct answers to speed up
    tmp_answers = {}
    for r in all_mapping:
        qid = r["question_id"]
        dfcol = resolved_for_qid.get(qid)
        if dfcol and dfcol in df.columns:
            tmp_answers[qid] = dfcol
    distinct_answers = set()
    for i, resp in df.iterrows():
        for qid, col in tmp_answers.items():
            a = clean(resp.get(col, ""))
            if a: distinct_answers.add(a)
    for t in distinct_answers:
        _ = embed_cached(t)

    for i, resp in df.iterrows():
        out = {}
        if pd.notna(dt_series.iloc[i]):
            out["ID"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["ID"] = str(i)
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr = {}
        ai_flags = []

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            dfcol = resolved_for_qid.get(qid)
            if not dfcol or dfcol not in df.columns:
                continue

            ans = clean(resp.get(dfcol, ""))
            if not ans:
                continue
            ai_flags.append(looks_ai_like(ans))
            vec = embed_cached(ans)

            # only Q1..Q4
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4):
                continue

            # sims
            sims_q = sims_a = sims_g = {}
            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)

            if vec is not None:
                if qkey and qkey in q_centroids:
                    sims_q = {s: cos_sim(vec, v) for s, v in q_centroids[qkey].items() if v is not None}
                if attr in attr_centroids:
                    sims_a = {s: cos_sim(vec, v) for s, v in attr_centroids[attr].items() if v is not None}
                sims_g = {s: cos_sim(vec, v) for s, v in global_centroids.items() if v is not None}

                band_raw, conf_raw, src_used = _pick_with_conf(sims_q, sims_a, sims_g)

                # QA overlap guard
                qtext = (by_qkey.get(qkey, {}) or {}).get("question_text", "")
                overlap = qa_overlap(ans, qtext or qhint)
                sc = band_raw
                conf = conf_raw
                if overlap < MIN_QA_OVERLAP:
                    sc = min(sc, 1)
                    conf = min(conf, 0.65)

                # global band for disagreement test
                band_global = max(sims_g, key=sims_g.get) if sims_g else None
                route, reason = _route(sc, conf, band_global, looks_ai_like(ans))

            else:
                sc, conf, route, reason = None, 0.0, "review", "No embedding"

            # write
            score_key   = f"{attr}_Qn{qn}"
            rubric_key  = f"{attr}_Rubric_Qn{qn}"
            conf_key    = f"{attr}_Qn{qn}_Confidence"
            route_key   = f"{attr}_Qn{qn}_Route"
            reason_key  = f"{attr}_Qn{qn}_RouteReason"

            if sc is None:
                out.setdefault(score_key, "")
                out.setdefault(rubric_key, "")
                out.setdefault(conf_key, "")
                out.setdefault(route_key, "review")
                out.setdefault(reason_key, "No signal")
            else:
                out[score_key]  = int(sc)
                out[rubric_key] = BANDS[int(sc)]
                out[conf_key]   = round(conf, 3)
                out[route_key]  = route
                out[reason_key] = reason
                per_attr.setdefault(attr, []).append(int(sc))

        # fill defaults for missing cols
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")
                out.setdefault(f"{attr}_Qn{qn}_Confidence", "")
                out.setdefault(f"{attr}_Qn{qn}_Route", "review")
                out.setdefault(f"{attr}_Qn{qn}_RouteReason", "")

        # attribute averages & overall
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
        out["AI_suspected"] = bool(any(ai_flags))
        rows_out.append(out)

    res_df = pd.DataFrame(rows_out)

    def order_cols(cols):
        ordered = ["ID","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [
                    f"{attr}_Qn{qn}",
                    f"{attr}_Rubric_Qn{qn}",
                    f"{attr}_Qn{qn}_Confidence",
                    f"{attr}_Qn{qn}_Route",
                    f"{attr}_Qn{qn}_RouteReason",
                ]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI_suspected"]
        extras = [c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols] + extras

    return res_df.reindex(columns=order_cols(list(res_df.columns)))

# ==============================
# SPLIT: autos vs review
# ==============================
def split_auto_review(scored_df: pd.DataFrame):
    q_route_cols = [c for c in scored_df.columns if c.endswith("_Route") and "_Qn" in c]
    def row_is_auto(row):
        vals = [str(row[c]).strip().lower() for c in q_route_cols if c in row.index]
        return bool(vals) and all(v == "auto" for v in vals if v)
    auto_mask = scored_df.apply(row_is_auto, axis=1)
    return scored_df[auto_mask].copy(), scored_df[~auto_mask].copy()

# ==============================
# EXPORTS
# ==============================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return bio.getvalue()

# ==============================
# Google Sheets
# ==============================
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

def push_sheet_tab(sh: gspread.Spreadsheet, title: str, df: pd.DataFrame):
    try:
        w = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        w = sh.add_worksheet(title=title, rows="20000", cols="200")
    header = df.columns.astype(str).tolist()
    values = df.astype(object).where(pd.notna(df), "").values.tolist()
    w.clear()
    col_end = _to_a1_col(len(header))
    sh.values_batch_update(body={
        "valueInputOption":"USER_ENTERED",
        "data":[{"range": f"'{title}'!A1:{col_end}{len(values)+1}",
                 "values":[header] + values}]
    })
    _post_write_formatting(w, len(header))

# ==============================
# OPTIONAL: star schema (for BI)
# ==============================
def build_star_schema_from_scored(scored: pd.DataFrame):
    cols = list(scored.columns)
    qn_score_cols = [c for c in cols if "_Qn" in c and not c.endswith(")")]
    avg_cols = [c for c in cols if c.endswith("_Avg (0–3)")]

    def attr_from_score_col(c): return c.split("_Qn")[0]
    attributes = sorted(set([attr_from_score_col(c) for c in qn_score_cols]) |
                        set([c.replace("_Avg (0–3)", "") for c in avg_cols]))

    qrows = []
    for c in qn_score_cols:
        attr = attr_from_score_col(c)
        try:
            qn = int(c.split("_Qn")[1])
        except:
            continue
        rubric_col = f"{attr}_Rubric_Qn{qn}"
        r = scored[["ID","Staff ID"]].copy()
        r["Attribute"] = attr
        r["QuestionNo"] = qn
        r["Score"] = scored[c]
        r["RubricBand"] = scored[rubric_col] if rubric_col in scored.columns else np.nan
        qrows.append(r)
    fact_question = pd.concat(qrows, ignore_index=True) if qrows else pd.DataFrame(
        columns=["ID","Staff ID","Attribute","QuestionNo","Score","RubricBand"]
    )

    arows = []
    for attr in attributes:
        avg_col = f"{attr}_Avg (0–3)"; rank_col = f"{attr}_RANK"
        r = scored[["ID","Staff ID"]].copy()
        r["Attribute"] = attr
        r["AvgScore"] = scored.get(avg_col)
        r["RankBand"] = scored.get(rank_col)
        arows.append(r)
    fact_attribute = pd.concat(arows, ignore_index=True) if arows else pd.DataFrame(
        columns=["ID","Staff ID","Attribute","AvgScore","RankBand"]
    )

    sub_cols = ["ID","Staff ID","Duration_min","Overall Total (0–24)","Overall Rank","AI_suspected"]
    for c in sub_cols:
        if c not in scored.columns: scored[c] = np.nan
    submission = scored[sub_cols].copy()

    def _to_dt(x):
        try: return pd.to_datetime(x)
        except: return pd.NaT
    submission["DateTimeUTC"] = submission["ID"].apply(_to_dt)
    submission["date_key"] = submission["DateTimeUTC"].dt.strftime("%Y%m%d").astype("Int64")

    dim_date = submission[["date_key","DateTimeUTC"]].dropna(subset=["date_key"]).drop_duplicates().copy()
    if not dim_date.empty:
        dt = pd.to_datetime(dim_date["DateTimeUTC"])
        dim_date["year"] = dt.dt.year
        dim_date["quarter"] = dt.dt.quarter
        dim_date["month"] = dt.dt.month
        dim_date["day"] = dt.dt.day
        dim_date["week"] = dt.dt.isocalendar().week.astype(int)
        dim_date["dow"] = dt.dt.dayofweek
        dim_date["month_name"] = dt.dt.month_name()
        dim_date["dow_name"] = dt.dt.day_name()

    dim_staff = submission[["Staff ID"]].rename(columns={"Staff ID":"staff_natural_key"}).drop_duplicates()
    dim_staff["staff_key"] = dim_staff["staff_natural_key"].astype("category").cat.codes + 1
    dim_staff = dim_staff[["staff_key","staff_natural_key"]]

    dim_attribute = pd.DataFrame({"attribute_name": attributes})
    if not dim_attribute.empty:
        dim_attribute["attribute_key"] = dim_attribute["attribute_name"].astype("category").cat.codes + 1
        dim_attribute = dim_attribute[["attribute_key","attribute_name"]]

    staff_map = dict(zip(dim_staff["staff_natural_key"], dim_staff["staff_key"]))
    attr_map = dict(zip(dim_attribute["attribute_name"], dim_attribute["attribute_key"]))

    submission["staff_key"] = submission["Staff ID"].map(staff_map)

    fact_attribute = (fact_attribute
        .assign(staff_key=fact_attribute["Staff ID"].map(staff_map),
                attribute_key=fact_attribute["Attribute"].map(attr_map))
        .merge(submission[["ID","date_key"]], on="ID", how="left")
        [["ID","date_key","staff_key","attribute_key","AvgScore","RankBand"]]
    )

    fact_question = (fact_question
        .assign(staff_key=fact_question["Staff ID"].map(staff_map),
                attribute_key=fact_question["Attribute"].map(attr_map))
        .merge(submission[["ID","date_key"]], on="ID", how="left")
        [["ID","date_key","staff_key","attribute_key","QuestionNo","Score","RubricBand"]]
    )

    submission_out = submission[["ID","date_key","staff_key","Duration_min",
                                 "Overall Total (0–24)","Overall Rank","AI_suspected"]]

    return {
        "fact_attribute": fact_attribute,
        "fact_question": fact_question,
        "dim_staff": dim_staff,
        "dim_attribute": dim_attribute,
        "dim_date": dim_date,
        "submission": submission_out,
    }

# ==============================
# UI / MAIN
# ==============================
def main():
    st.title("📊 Advisory Scoring: Kobo → Confidence Router → Google Sheets")

    # Controls: turn on hands-free mode via secrets
    AUTO_RUN  = bool(st.secrets.get("AUTO_RUN", False))
    AUTO_PUSH = bool(st.secrets.get("AUTO_PUSH", False))
    PUSH_STAR = bool(st.secrets.get("PUSH_STAR_SCHEMA", True))  # optional star-schema tabs

    def run_pipeline():
        try:
            mapping = load_mapping_from_path(MAPPING_PATH)
        except Exception as e:
            st.error(f"Failed to load mapping from {MAPPING_PATH}: {e}")
            st.stop()

        try:
            exemplars = read_jsonl_path(EXEMPLARS_PATH)
            if not exemplars:
                st.error(f"Exemplars file is empty: {EXEMPLARS_PATH}")
                st.stop()
        except Exception as e:
            st.error(f"Failed to read exemplars from {EXEMPLARS_PATH}: {e}")
            st.stop()

        with st.spinner("Building semantic centroids..."):
            q_centroids, attr_centroids, global_centroids, by_qkey, question_texts = build_centroids(exemplars)

        with st.spinner("Fetching Kobo submissions..."):
            df = fetch_kobo_dataframe()

        if df.empty:
            st.warning("No Kobo submissions found.")
            st.stop()

        st.caption("Fetched sample:")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("Scoring with confidence & routing..."):
            scored_df = score_dataframe(
                df, mapping, q_centroids, attr_centroids, global_centroids, by_qkey, question_texts
            )

        autos_df, review_df = split_auto_review(scored_df)

        st.success("✅ Scoring complete.")
        st.write(f"Auto-accepted rows: {len(autos_df)} · Needs review: {len(review_df)}")
        st.dataframe(scored_df.head(30), use_container_width=True)

        st.download_button(
            "⬇️ Download Excel (full scored)",
            data=to_excel_bytes(scored_df),
            file_name="Advisory_Scoring.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.session_state["scored_df"] = scored_df
        st.session_state["autos_df"]  = autos_df
        st.session_state["review_df"] = review_df

    # Auto run once if enabled
    if AUTO_RUN and not st.session_state.get("auto_ran_once"):
        st.session_state["auto_ran_once"] = True
        run_pipeline()

    # Manual run
    if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True):
        run_pipeline()

    # Sheets export section
    if "scored_df" in st.session_state and st.session_state["scored_df"] is not None:
        with st.expander("📤 Google Sheets export", expanded=True):
            st.write("Spreadsheet key:", st.secrets.get("GSHEETS_SPREADSHEET_KEY") or "⚠️ Not set")
            st.write("Worksheet name:", DEFAULT_WS_NAME)
            col1, col2, col3 = st.columns(3)

            def do_push():
                ws = _open_ws_by_key()
                sh = ws.spreadsheet
                push_sheet_tab(sh, DEFAULT_WS_NAME, st.session_state["scored_df"])
                push_sheet_tab(sh, "autos_accepted", st.session_state["autos_df"])
                push_sheet_tab(sh, "review_queue", st.session_state["review_df"])
                if PUSH_STAR:
                    st.info("Building star schema…")
                    tables = build_star_schema_from_scored(st.session_state["scored_df"])
                    for title, tdf in tables.items():
                        push_sheet_tab(sh, title, tdf)
                st.success("✅ Wrote scored, autos_accepted, review_queue" + (" + star schema" if PUSH_STAR else "") + " to Google Sheets.")

            with col1:
                if st.button("Push all to Google Sheets", use_container_width=True):
                    try:
                        do_push()
                    except Exception as e:
                        st.error(f"Push failed: {e}")

            with col2:
                st.caption(f"AUTO_RUN={bool(st.secrets.get('AUTO_RUN', False))}, AUTO_PUSH={bool(st.secrets.get('AUTO_PUSH', False))}")

            with col3:
                if AUTO_PUSH:
                    try:
                        do_push()
                    except Exception as e:
                        st.error(f"Auto-push failed: {e}")

if __name__ == "__main__":
    main()
