# app.py
import streamlit as st
st.set_page_config(page_title="Advisory Scoring (Kobo → Excel/Power BI)", layout="wide")
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json, re, unicodedata
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
import gspread

# ==============================
# CONSTANTS / PATHS
# ==============================
KOBO_BASE        = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID    = st.secrets.get("KOBO_ASSET_ID", "")
KOBO_TOKEN       = st.secrets.get("KOBO_TOKEN", "")


DATASETS_DIR     = Path("DATASETS")
MAPPING_PATH     = DATASETS_DIR / "mapping.csv"
EXEMPLARS_PATH   = DATASETS_DIR / "advisory_exemplars_smart.cleaned.jsonl"

BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 21, 24),
    ("Strategic Advisor",       16, 20),
    ("Emerging Advisor",        10, 15),
    ("Needs Capacity Support",   0,  9),
]

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

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

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

AI_RX = re.compile(r"(?:-{3,}|—{2,}|_{2,}|\.{4,}|as an ai\b|i am an ai\b)", re.I)
def looks_ai_like(t): return bool(AI_RX.search(clean(t)))

def kobo_url(asset_uid: str, kind: str = "submissions"):
    # kind ∈ {"submissions", "data"}
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

def normalize_col_name(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("’","'").replace("“","\"").replace("”","\"")
    s = re.sub(r"\s+"," ", s)
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    return s

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
# QUESTION_ID → KOBO COLUMN RESOLVER (critical)
# ==============================
# Map your question_id prefixes to Kobo section prefixes
QID_PREFIX_TO_SECTION = {
    "SAT": "A1",  # Strategic & analytical thinking
    "CT":  "A2",  # Credibility & trustworthiness
    "ECI": "A3",  # Effective communication & influence
    "CSF": "A4",  # Client & stakeholder focus
    "FCP": "A5",  # Fostering collaboration & partnership
    "ERI": "A6",  # Ensuring relevance & impact
    "SOA": "A7",  # Solution orientation & adaptability
    "CSE": "A8",  # Capacity strengthening & empowerment support
}

QNUM_RX = re.compile(r"_Q(\d+)$")

def build_kobo_base_from_qid(question_id: str) -> str | None:
    """
    SAT_Q1 -> A1_1 -> Advisory/A1_Section/A1_1
    """
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
    section = QID_PREFIX_TO_SECTION[prefix]  # e.g., A1
    return f"Advisory/{section}_Section/{section}_{qn}"

def expand_possible_kobo_columns(base: str) -> list[str]:
    """
    Likely variants that show up across different Kobo exports.
    """
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

# --- put this near your other helpers ---
def _score_kobo_header(col: str, token: str) -> int:
    """
    Heuristic score for how well a Kobo header 'col' matches 'token' (e.g., 'A1_5').
    Higher is better. This lets us tolerate different groupings/labels.
    """
    c = col.lower()
    t = token.lower()

    # exact token
    if c == t:
        return 100

    score = 0
    # strong: path ends with /token
    if c.endswith("/" + t):
        score = max(score, 95)
    # contains clear segment /token/
    if f"/{t}/" in c:
        score = max(score, 92)
    # contains token with separator (very common)
    if f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c:
        score = max(score, 90)
    # base contains token anywhere
    if t in c:
        score = max(score, 80)

    # little bonuses if it looks like label/text columns we want
    if "english" in c or "label" in c:   # human-readable labels
        score += 3
    if "answer (text)" in c or "answer_text" in c or "text" in c:
        score += 2

    # small preference for paths that look advisory-ish
    if "advisory/" in c or "/a" in c:
        score += 1

    return score

# --- REPLACE your resolve_kobo_column_for_mapping with this one ---
def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    """
    Robust resolver:
      1) derive token from question_id (e.g., SAT_Q5 -> 'A1_5')
      2) score ALL headers with heuristics and pick the best
      3) if nothing passes a safe threshold, fuzzy on prompt_hint
    """
    base = build_kobo_base_from_qid(question_id)  # Advisory/A1_Section/A1_5 (maybe right, maybe not)
    # regardless of base, compute the essential token 'A1_5'
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

    # 1) fast path: exact base matches (keep your old wins)
    if base and base in df_cols:
        return base
    if base:
        for v in expand_possible_kobo_columns(base):
            if v in df_cols:
                return v
        for c in df_cols:
            if c.startswith(base):
                return c

    # 2) heuristic scan over ALL headers using the token
    if token:
        best_col = None
        best_score = 0
        for col in df_cols:
            s = _score_kobo_header(col, token)
            if s > best_score:
                best_score = s
                best_col = col
        # choose if confident enough
        if best_col and best_score >= 82:
            return best_col

    # 3) final fallback: fuzzy on prompt_hint against headers
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
    by_qkey = {}
    by_attr = {}
    question_texts = []

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

def resolve_qkey(q_centroids, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in q_centroids:
        return qid
    hint = clean(prompt_hint or "")
    if not hint or not question_texts:
        return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted:
                return k
    return None

_embed_cache: dict[str, np.ndarray] = {}
def embed_cached(text: str):
    t = clean(text)
    if not t: return None
    if t in _embed_cache: return _embed_cache[t]
    vec = get_embedder().encode(t, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    _embed_cache[t] = vec
    return vec

# ==============================
# SCORING
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts):

    df_cols = list(df.columns)

    # show quick peek of likely advisory columns
    with st.expander("🔎 Debug: Advisory section columns present", expanded=False):
        sample_cols = [c for c in df_cols if "/A" in c or "Advisory/" in c or c.startswith("A")]
        st.write(sample_cols[:80])

    # Staff ID / time fields
    staff_id_col = next((c for c in df.columns if c.strip().lower() == "staff id" or c.strip().lower()=="staff_id"), None)
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

    # Pre-resolve Kobo columns for each mapping row using question_id (critical change)
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

    # Row-wise scoring
    for i, resp in df.iterrows():
        out = {}
        # ID
        if pd.notna(dt_series.iloc[i]):
            out["ID"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["ID"] = str(i)
        # Staff ID
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        # Duration
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr = {}
        ai_flags = []

        # warm embedding cache for this row
        # collect answers first
        tmp_answers = {}
        for r in all_mapping:
            qid = r["question_id"]
            dfcol = resolved_for_qid.get(qid)
            if dfcol and dfcol in df.columns:
                tmp_answers[qid] = clean(resp.get(dfcol, ""))
        for t in set(tmp_answers.values()):
            if t: _ = embed_cached(t)

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

            # resolve exemplar key for this question
            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)

            sc = None
            if vec is not None:
                # question-level
                if qkey and qkey in q_centroids:
                    sims = {s: cos_sim(vec, v) for s, v in q_centroids[qkey].items() if v is not None}
                    if sims:
                        sc = max(sims, key=sims.get)
                        qtext = by_qkey[qkey]["question_text"]
                        if qa_overlap(ans, qtext or qhint) < MIN_QA_OVERLAP:
                            sc = min(sc, 1)
                # attribute-level
                if sc is None and attr in attr_centroids:
                    sims = {s: cos_sim(vec, v) for s, v in attr_centroids[attr].items() if v is not None}
                    if sims:
                        sc = max(sims, key=sims.get)
                        if qa_overlap(ans, qhint) < MIN_QA_OVERLAP:
                            sc = min(sc, 1)
                # global fallback
                if sc is None:
                    sims = {s: cos_sim(vec, v) for s, v in global_centroids.items() if v is not None}
                    if sims:
                        sc = max(sims, key=sims.get)
                        if qa_overlap(ans, qhint) < MIN_QA_OVERLAP:
                            sc = min(sc, 1)

            # figure Qn index
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4):
                continue

            score_key  = f"{attr}_Qn{qn}"
            rubric_key = f"{attr}_Rubric_Qn{qn}"
            if sc is None:
                out.setdefault(score_key, "")
                out.setdefault(rubric_key, "")
            else:
                out[score_key]  = int(sc)
                out[rubric_key] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

        # fill missing structure
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # per-attribute averages & overall
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
    
    # order columns exactly
    def order_cols(cols):
        ordered = ["ID","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI_suspected"]
        extras = [c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols] + extras

    return res_df.reindex(columns=order_cols(list(res_df.columns)))

# ==============================
# EXPORTS
# ==============================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return bio.getvalue()

def _sanitize_for_pbi(df: pd.DataFrame) -> pd.DataFrame:
    # Power BI chokes on NaN and very long strings; ensure str cols are trimmed and NaN -> None
    out = df.copy()

    # Convert numpy types and NaN to Python/None-friendly values
    def _clean_cell(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.bool_,)):  # convert numpy bools
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            # Power BI accepts floats; keep them, but ensure not NaN (handled above)
            return float(v)
        # strings: trim overly long values (PBI has row & cell limits)
        s = str(v)
        return s[:4000]

    for c in out.columns:
        out[c] = out[c].map(_clean_cell)

    # Power BI is fine with various column names, but just in case, strip control chars
    safe_cols = []
    for c in out.columns:
        name = re.sub(r"[\x00-\x1f]+", "", str(c)).strip()
        safe_cols.append(name[:120])  # keep names sane
    out.columns = safe_cols

    return out


def _detect_rows_wrapper(url: str) -> bool:
    # If your URL already ends with /rows or ?key=…/rows etc., Power BI expects {"rows": [...]}
    return "/rows" in url.lower()


def _chunk(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

GS_SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

def _get_sa_dict_from_secrets() -> dict:
    if "gcp_service_account" not in st.secrets:
        raise ValueError("Add [gcp_service_account] to .streamlit/secrets.toml")
    sa = dict(st.secrets["gcp_service_account"])
    # fix escaped newlines
    if isinstance(sa.get("private_key"), str):
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    return sa

@st.cache_resource(show_spinner=False)
def _gs_client():
    sa_dict = _get_sa_dict_from_secrets()
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_dict, GS_SCOPE)
    return gspread.authorize(creds)

def _open_ws_by_name():
    ss_name = st.secrets.get("SPREADSHEET_NAME")
    ws_name = st.secrets.get("SHEET_NAME")
    if not ss_name or not ws_name:
        raise ValueError("Set SPREADSHEET_NAME and SHEET_NAME in secrets.toml")
    gc = _gs_client()
    sh = gc.open(ss_name)  # open by NAME; use gc.open_by_key(...) if you prefer
    try:
        return sh.worksheet(ws_name)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=ws_name, rows="1000", cols="200")

def upload_df_to_gsheets(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        ws = _open_ws_by_name()
        values = [df.columns.astype(str).tolist()] + df.astype(object).where(pd.notna(df), "").values.tolist()
        ws.clear()
        ws.update(values)
        return True, f"Uploaded {len(df)} rows to '{st.secrets['SPREADSHEET_NAME']}/{st.secrets['SHEET_NAME']}' ✅"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# ==============================
# UI
# ==============================
st.title("📊 Advisory Scoring: Kobo → Scored Excel (and Power BI)")

if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True):
    # Load mapping + exemplars
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

    with st.spinner("Scoring responses..."):
        scored_df = score_dataframe(df, mapping, q_centroids, attr_centroids, global_centroids, by_qkey, question_texts)

    st.success("✅ Scoring complete.")
    st.dataframe(scored_df.head(50), use_container_width=True)
    # 🔐 keep it for the Power BI button (survives reruns)
    st.session_state["scored_df"] = scored_df
    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(scored_df),
        file_name="Individual_Advisory_Scoring_Sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# Push to Google Sheets (after scoring)
if "scored_df" in st.session_state:
    with st.expander("Google Sheets export", expanded=True):
        st.write("Spreadsheet:", st.secrets.get("SPREADSHEET_NAME", "⚠️ not set"))
        st.write("Worksheet:", st.secrets.get("SHEET_NAME", "⚠️ not set"))
        if st.button("📤 Send scored table to Google Sheets", use_container_width=True):
            ok, msg = upload_df_to_gsheets(st.session_state["scored_df"])
            st.success(msg) if ok else st.error(msg)


st.markdown("---")
st.caption("Derives Kobo columns from question_id (e.g., SAT_Q1 → Advisory/A1_Section/A1_1), falls back to fuzzy on prompt_hint, then scores via SBERT centroids.") 
