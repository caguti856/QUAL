# app.py
import streamlit as st
st.set_page_config(page_title="Advisory Scoring (Kobo → Excel/Power BI)", layout="wide")

import json, re, unicodedata
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# ==============================
# CONSTANTS / PATHS
# ==============================
KOBO_BASE       = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID   = st.secrets.get("KOBO_ASSET_ID", "")
KOBO_TOKEN      = st.secrets.get("KOBO_TOKEN", "")
POWERBI_PUSH_URL= st.secrets.get("POWERBI_PUSH_URL", "")

def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

DATASETS_DIR = Path("DATASETS")
MAP_PATH     = DATASETS_DIR / "mapping.csv"  # your file (no extension). Recommend: rename to mapping.csv
EX_PATH      = DATASETS_DIR / "advisory_exemplars_smart.cleaned.jsonl"

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
    ("Strategic Advisor",       16, 20),
    ("Emerging Advisor",        10, 15),
    ("Needs Capacity Support",   0,  9),
]

# ---------- tiny helpers ----------
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

# ==============================
# LOADERS
# ==============================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID or not KOBO_TOKEN:
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
        except requests.HTTPError:
            if r.status_code in (401, 403):
                st.error("Kobo auth failed: check KOBO_TOKEN / tenant.")
                return pd.DataFrame()
            if r.status_code == 404:
                continue
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    try:
        m = pd.read_csv(path, engine="python")
    except Exception:
        m = pd.read_excel(path)
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
            if line.strip(): rows.append(json.loads(line))
    return rows

# ==============================
# ROBUST COLUMN RESOLUTION
# ==============================
def normalize_col_name(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("’","'").replace("“","\"").replace("”","\"")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    return s

def build_resolution_map(df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[dict, list]:
    df_col_index = {c: normalize_col_name(c) for c in df.columns}
    norm_to_dfcols = {}
    for orig, norm in df_col_index.items():
        norm_to_dfcols.setdefault(norm, []).append(orig)
    all_df_norms = list(norm_to_dfcols.keys())

    def resolve_df_column(wanted: str) -> str | None:
        if wanted in df.columns:
            return wanted
        w_norm = normalize_col_name(wanted)
        if w_norm in norm_to_dfcols:
            return norm_to_dfcols[w_norm][0]
        # startswith/contains
        for norm_key in all_df_norms:
            if norm_key.startswith(w_norm) or w_norm.startswith(norm_key):
                return norm_to_dfcols[norm_key][0]
        # fuzzy
        match = process.extractOne(w_norm, all_df_norms, scorer=fuzz.token_set_ratio)
        if match and match[1] >= 90:
            return norm_to_dfcols[match[0]][0]
        return None

    resolved_map, unresolved = {}, []
    for col in mapping["column"]:
        resolved = resolve_df_column(col)
        if resolved:
            resolved_map[col] = resolved
        else:
            unresolved.append(col)
    return resolved_map, unresolved

# ==============================
# EMBEDDINGS & CENTROIDS
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
        return embs.mean(axis=0)

    # question centroids 0..3
    q_centroids = {}
    for k,v in by_qkey.items():
        buckets = {0:[],1:[],2:[],3:[]}
        for t,s in zip(v["texts"], v["scores"]):
            if t: buckets[int(s)].append(t)
        q_centroids[k] = {sc: centroid(txts) for sc, txts in buckets.items()}

    # attribute centroids
    attr_centroids = {attr: {sc: centroid(txts) for sc, txts in bucks.items()} for attr, bucks in by_attr.items()}

    # global centroids
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

def cos_best(vec, centroid_dict):
    sims = {s: cos_sim(vec, v) for s, v in centroid_dict.items() if v is not None}
    if not sims: return None
    return int(max(sims, key=sims.get))

# ==============================
# SCORING
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids, by_qkey, question_texts):

    # 0) Resolve mapping columns → df columns
    resolved_map, unresolved = build_resolution_map(df, mapping)
    with st.expander("🔎 Column resolution details"):
        st.write(f"Resolved: {len(resolved_map)} | Unresolved: {len(unresolved)}")
        if resolved_map:
            st.dataframe(pd.DataFrame([{"mapping.column": k, "df.column": v} for k,v in resolved_map.items()]))
        if unresolved:
            st.warning("Unresolved mapping columns (fix these names in DATASETS/mapping_csv):")
            st.write(unresolved[:50])

    # 1) Staff ID / Date / Duration
    staff_id_col = next((c for c in df.columns if normalize_col_name(c) in ("staff id","staff_id")), None)
    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end","_submission_time","SubmissionDate","submissiondate"] if c in df.columns), None)

    dt_series = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series([pd.NaT]*len(df))
    start_dt  = pd.to_datetime(df[start_col], errors="coerce") if start_col else pd.Series([pd.NaT]*len(df))
    end_dt    = pd.to_datetime(df[end_col], errors="coerce")   if end_col   else pd.Series([pd.NaT]*len(df))
    duration_min = ((end_dt - start_dt).dt.total_seconds()/60.0).round(2)

    # 2) Score
    rows_out = []
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

    for i, resp in df.iterrows():
        # warm embed (batch unique answers using resolved df columns)
        needed_dfcols = []
        for r in all_mapping:
            dfcol = resolved_map.get(r["column"])
            if dfcol and dfcol in df.columns:
                needed_dfcols.append(dfcol)

        raw_answers = {col: clean(resp.get(col, "")) for col in needed_dfcols}
        uniq_texts = list({t for t in raw_answers.values() if t})
        _ = [embed_cached(t) for t in uniq_texts]

        out = {}
        # ID from date col if parseable
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
            dfcol = resolved_map.get(col)
            if not dfcol or dfcol not in df.columns:
                continue

            ans = raw_answers.get(dfcol, "")
            ai_flags.append(looks_ai_like(ans))

            vec = embed_cached(ans)
            if vec is None:
                continue

            # Resolve qkey from exemplars (qid first, else fuzzy on prompt_hint)
            qkey = (qid or "").strip()
            if not qkey or qkey not in q_centroids:
                qtexts = [by_qkey[k]["question_text"] for k in by_qkey if by_qkey[k]["question_text"]]
                match = process.extractOne(clean(qhint or ""), qtexts, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 80:
                    wanted = match[0]
                    for k in by_qkey:
                        if clean(by_qkey[k]["question_text"]) == wanted:
                            qkey = k
                            break
            sc = None

            # 1) question-level centroid
            if qkey and qkey in q_centroids:
                sc = cos_best(vec, q_centroids[qkey])

            # 2) attribute-level fallback
            if sc is None and attr in attr_centroids:
                sc = cos_best(vec, attr_centroids[attr])

            # 3) global-level fallback
            if sc is None:
                sc = cos_best(vec, global_centroids)

            if sc is None:
                continue

            # only Q1..Q4
            qn = None
            if "_Q" in qid:
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4):
                continue

            out[f"{attr}_Qn{qn}"] = sc
            out[f"{attr}_Rubric_Qn{qn}"] = BANDS[sc]
            per_attr.setdefault(attr, []).append(int(sc))

        # fill blanks to keep all columns present
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # per-attr summaries
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

    # final column order
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
    try:
        r = requests.post(POWERBI_PUSH_URL, json=send_df.to_dict(orient="records"), timeout=60)
        if r.status_code in (200, 202):
            return True, "Success"
        return False, f"{r.status_code} {r.text[:300]}"
    except Exception as e:
        return False, str(e)

# ==============================
# UI
# ==============================
st.title("📊 Advisory Scoring: Kobo → Scored Excel (and Power BI)")

run_btn = st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True)

if run_btn:
    # mapping + exemplars
    try:
        mapping = load_mapping_from_path(MAP_PATH)
    except Exception as e:
        st.error(f"Failed to load mapping from {MAP_PATH}: {e}")
        st.stop()
    try:
        exemplars = read_jsonl_path(EX_PATH)
        if not exemplars:
            st.error(f"Exemplars file is empty: {EX_PATH}")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read exemplars from {EX_PATH}: {e}")
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

    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(scored_df),
        file_name="Individual_Advisory_Scoring_Sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    if POWERBI_PUSH_URL:
        if st.button("📤 Push to Power BI", use_container_width=True):
            ok, msg = push_to_powerbi(scored_df)
            if ok: st.success("Pushed to Power BI.")
            else:  st.error(f"Failed to push: {msg}")

