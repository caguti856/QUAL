import os, json, re, unicodedata, io, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Advisory Scoring (Kobo → Excel/Power BI)", layout="wide")

# 👉 Put these in your Streamlit Secrets:
# st.secrets["KOBO_TOKEN"] = "xxxxx"
# st.secrets["KOBO_ASSET_ID"] = "atdspJQv7RBwjkmaVFRS43"
# st.secrets["POWERBI_PUSH_URL"] = "https://api.powerbi.com/beta/.../rows?key=..."

KOBO_TOKEN = st.secrets.get("KOBO_TOKEN", "")
KOBO_ASSET_ID = st.secrets.get("KOBO_ASSET_ID", "")
POWERBI_PUSH_URL = st.secrets.get("POWERBI_PUSH_URL", "")

if not KOBO_TOKEN or not KOBO_ASSET_ID:
    st.warning("Please set KOBO_TOKEN and KOBO_ASSET_ID in `st.secrets`.")

KOBO_API_URL = (
    f"https://kf.kobotoolbox.org/assets/{KOBO_ASSET_ID}/submissions/?format=json"
)

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

# ================================
# HELPERS
# ================================
def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def try_dt(x):
    if pd.isna(x): return None
    if isinstance(x, (pd.Timestamp, datetime)): return pd.to_datetime(x)
    try:
        return pd.to_datetime(str(x), errors="coerce")
    except Exception:
        return None

def minutes_between(start, end):
    if start is None or end is None: return ""
    try: return round((end - start).total_seconds()/60.0, 2)
    except: return ""

def cos_sim(a, b):
    if a is None or b is None: return -1e9
    return float(np.dot(a, b))

# very light “AI-ish” artifacts (non-blocking flag)
AI_RX = re.compile(r"(?:-{3,}|—{2,}|_{2,}|\.{4,}|as an ai\b|i am an ai\b)", re.I)
def looks_ai_like(t): return bool(AI_RX.search(clean(t)))

# ================================
# DATA FETCH / LOAD
# ================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo() -> pd.DataFrame:
    headers = {"Authorization": f"Token {KOBO_TOKEN}"} if KOBO_TOKEN else {}
    r = requests.get(KOBO_API_URL, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    results = data if isinstance(data, list) else data.get("results", [])
    df = pd.DataFrame(results)
    if not df.empty:
        df.columns = [str(c).strip() for c in df.columns]
    return df

def load_mapping(file_or_df) -> pd.DataFrame:
    if isinstance(file_or_df, pd.DataFrame):
        m = file_or_df.copy()
    else:
        if file_or_df.name.lower().endswith(".csv"):
            m = pd.read_csv(file_or_df)
        else:
            m = pd.read_excel(file_or_df)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), \
        "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    # keep only attributes we score
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def read_jsonl(file) -> list[dict]:
    rows = []
    for line in file:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if str(line).strip():
            rows.append(json.loads(line))
    return rows

# ================================
# MODELS & CENTROIDS
# ================================
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

# ================================
# SCORING RUN
# ================================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame, centroids, by_qkey):
    # detect Staff ID and time columns
    staff_id_col = next((c for c in df.columns if c.strip().lower() == "staff id"), None)
    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end","_submission_time","SubmissionDate","submissiondate"] if c in df.columns), None)

    dt_series = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series([pd.NaT]*len(df))
    if start_col: start_dt = pd.to_datetime(df[start_col], errors="coerce")
    else: start_dt = pd.Series([pd.NaT]*len(df))
    if end_col: end_dt = pd.to_datetime(df[end_col], errors="coerce")
    else: end_dt = pd.Series([pd.NaT]*len(df))
    duration_min = ((end_dt - start_dt).dt.total_seconds()/60.0).round(2)

    rows_out = []
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

    for i, resp in df.iterrows():
        # batch embed unique answers in this row
        needed_cols = [r["column"] for r in all_mapping if r["column"] in df.columns]
        raw_answers = {col: clean(resp.get(col, "")) for col in needed_cols}
        uniq_texts = list({t for t in raw_answers.values() if t})
        if uniq_texts:
            # prime cache (fast if already there)
            _ = [embed_cached(t) for t in uniq_texts]

        out = {}
        # ID: formatted date/time if parseable, else raw
        if pd.notna(dt_series.iloc[i]):
            out["ID"] = pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["ID"] = str(resp.get(date_col, resp.index))
        # Staff ID
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        # Duration
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr = {}
        ai_flags = []

        for r in all_mapping:
            col, qid, attr, qhint = r["column"], r["question_id"], r["attribute"], r.get("prompt_hint","")
            if col not in df.columns: 
                continue
            ans = raw_answers.get(col, "")
            ai_flags.append(looks_ai_like(ans))
            qkey = resolve_qkey(centroids, by_qkey, [], qid, qhint)  # question_texts not needed now
            if not qkey: 
                # fallback: try prompt_hint only if necessary
                qkey = resolve_qkey(centroids, by_qkey, [by_qkey[k]["question_text"] for k in by_qkey if by_qkey[k]["question_text"]], qid, qhint)
            if not qkey:
                continue

            sc = score_answer(ans, qkey, centroids)
            if sc is None: 
                continue

            # figure Qn index
            qn = None
            if "_Q" in qid:
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4):
                continue

            out[f"{attr}_Qn{qn}"] = sc
            out[f"{attr}_Rubric_Qn{qn}"] = BANDS[sc]
            per_attr.setdefault(attr, []).append(int(sc))

        # fill any missing Qn slots
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

    # enforce your exact column order
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
    import openpyxl
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return bio.getvalue()

def push_to_powerbi(df: pd.DataFrame) -> tuple[bool, str]:
    if not POWERBI_PUSH_URL:
        return False, "POWERBI_PUSH_URL not set in secrets."
    # Basic trimming to be safe with Power BI API limits
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

# ================================
# UI
# ================================
st.title("📊 Advisory Scoring: Kobo → Scored Excel (and Power BI)")

colA, colB = st.columns([1,1])
with colA:
    mapping_file = st.file_uploader("Upload mapping.csv", type=["csv","xlsx"])
with colB:
    exemplars_file = st.file_uploader("Upload exemplars JSONL", type=["jsonl"])

run_btn = st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True)

if run_btn:
    if not mapping_file or not exemplars_file:
        st.error("Please upload both mapping and exemplars JSONL.")
        st.stop()

    try:
        mapping = load_mapping(mapping_file)
    except Exception as e:
        st.error(f"Failed to load mapping: {e}")
        st.stop()

    # Read exemplars JSONL
    try:
        exemplars = read_jsonl(exemplars_file)
        if not exemplars:
            st.error("Exemplars file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read exemplars: {e}")
        st.stop()

    with st.spinner("Building semantic centroids..."):
        centroids, by_qkey, qtexts = build_centroids(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        try:
            df = fetch_kobo()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            st.stop()

    if df.empty:
        st.warning("No submissions found.")
        st.stop()

    st.caption("Fetched sample:")
    st.dataframe(df.head(), use_container_width=True)

    with st.spinner("Scoring responses..."):
        scored_df = score_dataframe(df, mapping, centroids, by_qkey)

    st.success("✅ Scoring complete.")
    st.dataframe(scored_df.head(50), use_container_width=True)

    # Download button
    xlsx_bytes = to_excel_bytes(scored_df)
    st.download_button(
        "⬇️ Download Excel",
        data=xlsx_bytes,
        file_name="Individual_Advisory_Scoring_Sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # Optional push to Power BI
    if POWERBI_PUSH_URL:
        if st.button("📤 Push to Power BI", use_container_width=True):
            ok, msg = push_to_powerbi(scored_df)
            if ok:
                st.success("Pushed to Power BI.")
            else:
                st.error(f"Failed to push: {msg}")

st.markdown("---")
st.caption("Tip: The app uses SBERT centroids (per score 0–3) and resolves mapping rows by `question_id` first, then fuzzy matches `prompt_hint` to exemplar `question_text` if needed.")
