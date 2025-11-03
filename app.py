import streamlit as st
st.set_page_config(page_title="Advisory Scoring (Kobo → Excel/Power BI)", layout="wide")

import os, json, re, unicodedata, io, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------
KOBO_BASE     = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID = st.secrets.get("KOBO_ASSET_ID", "")  # e.g. aSYXhRYB...
KOBO_TOKEN    = st.secrets.get("KOBO_TOKEN", "")
POWERBI_PUSH_URL = st.secrets.get("POWERBI_PUSH_URL", "")

def kobo_url(asset_uid: str, kind: str = "submissions"):
    # kind: "submissions" or "data"
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

DATASETS_DIR = Path("DATASETS")
DEFAULT_MAPPING_PATH   = DATASETS_DIR / "mapping.csv"
DEFAULT_EXEMPLARS_PATH = DATASETS_DIR / "advisory_exemplars_smart.cleaned.jsonl"

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
BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 21, 24),
    ("Strategic Advisor", 16, 20),
    ("Emerging Advisor", 10, 15),
    ("Needs Capacity Support", 0, 9),
]

# -----------------------------
# HELPERS
# -----------------------------
def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def try_dt(x):
    if pd.isna(x): return None
    if isinstance(x, (pd.Timestamp, datetime)): return pd.to_datetime(x)
    try: return pd.to_datetime(str(x), errors="coerce")
    except Exception: return None

def minutes_between(start, end):
    if start is None or end is None: return ""
    try: return round((end - start).total_seconds()/60.0, 2)
    except: return ""

def cos_sim(a, b):
    if a is None or b is None: return -1e9
    return float(np.dot(a, b))

AI_RX = re.compile(r"(?:-{3,}|—{2,}|_{2,}|\.{4,}|as an ai\b|i am an ai\b)", re.I)
def looks_ai_like(t): return bool(AI_RX.search(clean(t)))

# -----------------------------
# LOADERS
# -----------------------------
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
                continue  # try the other endpoint
            r.raise_for_status()
            payload = r.json()
            results = payload if isinstance(payload, list) else payload.get("results", [])
            if not results and "results" not in payload:
                results = payload  # some tenants return a bare array
            return pd.DataFrame(results)
        except requests.HTTPError:
            if r.status_code in (401, 403):
                st.error("Kobo auth failed: check KOBO_TOKEN and tenant (KOBO_BASE).")
                return pd.DataFrame()
            if r.status_code == 404:
                continue
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()

    st.error("Could not fetch data. Check KOBO_BASE, KOBO_ASSET_ID, and token permissions.")
    return pd.DataFrame()

def load_mapping_from_filelike(file_or_df) -> pd.DataFrame:
    if isinstance(file_or_df, pd.DataFrame):
        m = file_or_df.copy()
    else:
        name = getattr(file_or_df, "name", "")
        if str(name).lower().endswith(".csv"):
            m = pd.read_csv(file_or_df)
        else:
            m = pd.read_excel(file_or_df)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), \
        "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    if path.suffix.lower() == ".csv":
        m = pd.read_csv(path)
    else:
        m = pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), \
        "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def read_jsonl_filelike(file) -> list[dict]:
    rows = []
    for line in file:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if str(line).strip():
            rows.append(json.loads(line))
    return rows

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# -----------------------------
# EMBEDDINGS / CENTROIDS
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_centroids(exemplars: list[dict]):
    by_qkey = {}
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

    embedder = get_embedder()

    def build_centroids_for_q(texts, scores):
        out = {0:None,1:None,2:None,3:None}
        if not texts: return out
        embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        for sc in (0,1,2,3):
            idxs = [i for i,s in enumerate(scores) if s == sc]
            if idxs:
                out[sc] = embs[idxs].mean(axis=0)
        return out

    centroids = {k: build_centroids_for_q(v["texts"], v["scores"]) for k, v in by_qkey.items()}
    return centroids, by_qkey, question_texts

def resolve_qkey(centroids, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in centroids:
        return qid
    hint = clean(prompt_hint or "")
    if not hint or not question_texts:
        return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= 80:
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

def score_answer(ans_text: str, qkey: str, centroids: dict) -> int | None:
    if not ans_text or qkey not in centroids: return None
    vec = embed_cached(ans_text)
    if vec is None: return None
    sims = {sc: cos_sim(vec, v) for sc, v in centroids[qkey].items() if v is not None}
    if not sims: return None
    return int(max(sims, key=sims.get))

# -----------------------------
# SCORING
# -----------------------------
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame, centroids, by_qkey, question_texts):
    staff_id_col = next((c for c in df.columns if c.strip().lower() == "staff id"), None)

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

    for i, resp in df.iterrows():
        needed_cols = [r["column"] for r in all_mapping if r["column"] in df.columns]
        raw_answers = {col: clean(resp.get(col, "")) for col in needed_cols}
        uniq_texts = list({t for t in raw_answers.values() if t})
        if uniq_texts:
            _ = [embed_cached(t) for t in uniq_texts]  # warm cache

        out = {}
        # ID (date-time if parseable; else raw)
        if pd.notna(dt_series.iloc[i]):
            out["ID"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["ID"] = str(resp.get(date_col, resp.index))

        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr = {}
        ai_flags = []

        for r in all_mapping:
            col, qid, attr, qhint = r["column"], r["question_id"], r["attribute"], r.get("prompt_hint","")
            if col not in df.columns:
                continue
            ans = raw_answers.get(col, "")
            ai_flags.append(looks_ai_like(ans))

            qkey = resolve_qkey(centroids, by_qkey, question_texts, qid, qhint)
            if not qkey:
                continue

            sc = score_answer(ans, qkey, centroids)
            if sc is None:
                continue

            qn = None
            if "_Q" in qid:
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4):
                continue

            out[f"{attr}_Qn{qn}"] = sc
            out[f"{attr}_Rubric_Qn{qn}"] = BANDS[sc]
            per_attr.setdefault(attr, []).append(int(sc))

        # fill missing slots
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                out[f"{attr}_Avg (0–3)"] = ""
                out[f"{attr}_RANK"] = ""
            else:
                avg = float(np.mean(scores)); band = int(round(avg))
                overall_total += band
                out[f"{attr}_Avg (0–3)"] = round(avg, 2)
                out[f"{attr}_RANK"] = BANDS[band]

        out["Overall Total (0–24)"] = overall_total
        out["Overall Rank"] = next((label for label, lo, hi in OVERALL_BANDS if lo <= overall_total <= hi), "")
        out["AI_suspected"] = bool(any(ai_flags))
        rows_out.append(out)

    res_df = pd.DataFrame(rows_out)

    # exact column order
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

    res_df = res_df.reindex(columns=order_cols(list(res_df.columns)))
    return res_df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return bio.getvalue()

def push_to_powerbi(df: pd.DataFrame) -> tuple[bool, str]:
    if not POWERBI_PUSH_URL:
        return False, "POWERBI_PUSH_URL not set in secrets."
    send_df = df.copy()
    for c in send_df.columns:
        if send_df[c].dtype == object:
            send_df[c] = send_df[c].astype(str).str.slice(0, 4000)
    data_json = send_df.to_dict(orient="records")
    try:
        r = requests.post(POWERBI_PUSH_URL, json=data_json, timeout=60)
        if r.status_code in (200, 202):
            return True, "Success"
        return False, f"{r.status_code} {r.text[:300]}"
    except Exception as e:
        return False, str(e)

# -----------------------------
# UI
# -----------------------------
st.title("📊 Advisory Scoring: Kobo → Scored Excel (and Power BI)")

st.sidebar.subheader("Inputs")
use_datasets = st.sidebar.toggle("Use DATASETS folder", value=True)
mapping_file   = None
exemplars_file = None

if not use_datasets:
    mapping_file   = st.sidebar.file_uploader("Upload mapping.csv (or .xlsx)", type=["csv","xlsx"])
    exemplars_file = st.sidebar.file_uploader("Upload exemplars JSONL", type=["jsonl"])

run_btn = st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True)

if run_btn:
    # Load mapping
    try:
        if use_datasets:
            mapping = load_mapping_from_path(DEFAULT_MAPPING_PATH)
        else:
            if not mapping_file:
                st.error("Please provide mapping.")
                st.stop()
            mapping = load_mapping_from_filelike(mapping_file)
    except Exception as e:
        st.error(f"Failed to load mapping: {e}")
        st.stop()

    # Load exemplars
    try:
        if use_datasets:
            exemplars = read_jsonl_path(DEFAULT_EXEMPLARS_PATH)
        else:
            if not exemplars_file:
                st.error("Please provide exemplars JSONL.")
                st.stop()
            exemplars = read_jsonl_filelike(exemplars_file)
        if not exemplars:
            st.error("Exemplars file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read exemplars: {e}")
        st.stop()

    with st.spinner("Building semantic centroids..."):
        centroids, by_qkey, question_texts = build_centroids(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()

    if df.empty:
        st.warning("No Kobo submissions found.")
        st.stop()

    st.caption("Fetched sample:")
    st.dataframe(df.head(), use_container_width=True)

    with st.spinner("Scoring responses..."):
        scored_df = score_dataframe(df, mapping, centroids, by_qkey, question_texts)

    st.success("✅ Scoring complete.")
    st.dataframe(scored_df.head(50), use_container_width=True)

    # Download
    xlsx_bytes = to_excel_bytes(scored_df)
    st.download_button(
        "⬇️ Download Excel",
        data=xlsx_bytes,
        file_name="Individual_Advisory_Scoring_Sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # Push to Power BI
    if POWERBI_PUSH_URL:
        if st.button("📤 Push to Power BI", use_container_width=True):
            ok, msg = push_to_powerbi(scored_df)
            if ok:
                st.success("Pushed to Power BI.")
            else:
                st.error(f"Failed to push: {msg}")

st.markdown("---")
st.caption("SBERT centroids per score (0–3). Mapping rows resolve by `question_id` first, then fuzzy-match `prompt_hint` to exemplar `question_text`.")
