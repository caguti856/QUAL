# app.py
import streamlit as st
st.set_page_config(page_title="Advisory Scoring (Kobo → Excel/Power BI)", layout="wide")

import json, re, unicodedata
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# ==============================
# CONSTANTS / PATHS
# ==============================
KOBO_BASE        = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID    = st.secrets.get("KOBO_ASSET_ID", "")
KOBO_TOKEN       = st.secrets.get("KOBO_TOKEN", "")
POWERBI_PUSH_URL = st.secrets.get("POWERBI_PUSH_URL", "")

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
# COLUMN RESOLUTION (critical fix)
# ==============================
def build_resolution_map(df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[dict, list]:
    df_cols = list(df.columns)
    df_norm = {c: normalize_col_name(c) for c in df_cols}
    norm_to_orig = {}
    for orig, norm in df_norm.items():
        norm_to_orig.setdefault(norm, []).append(orig)
    all_norms = list(norm_to_orig.keys())

    def resolve_one(wanted: str) -> str | None:
        # exact
        if wanted in df_cols: return wanted
        w_norm = normalize_col_name(wanted)
        if w_norm in norm_to_orig: return norm_to_orig[w_norm][0]
        # contains/startswith
        for n in all_norms:
            if n.startswith(w_norm) or w_norm.startswith(n):
                return norm_to_orig[n][0]
        # fuzzy
        match = process.extractOne(w_norm, all_norms, scorer=fuzz.token_set_ratio)
        if match and match[1] >= 90:
            return norm_to_orig[match[0]][0]
        return None

    resolved, unresolved = {}, []
    for col in mapping["column"]:
        hit = resolve_one(col)
        if hit: resolved[col] = hit
        else:   unresolved.append(col)
    return resolved, unresolved

# ==============================
# SCORING
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts):
    # 1) map mapping["column"] -> df columns (robust)
    resolved_map, unresolved = build_resolution_map(df, mapping)

    # diagnostics panel
    with st.expander("🔎 Column resolution details", expanded=False):
        if resolved_map:
            st.write("Resolved mapping → Kobo columns (sample):")
            st.dataframe(pd.DataFrame(list(resolved_map.items()), columns=["mapping.column","kobo.column"]).head(50))
        if unresolved:
            st.warning(f"{len(unresolved)} mapping columns not found in Kobo data. They will be skipped.")
            st.write(unresolved[:50])

    # 2) detect Staff ID & times
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

    # 3) iterate rows and score
    for i, resp in df.iterrows():
        # Collect answers using resolved columns
        needed_pairs = [(r["column"], resolved_map.get(r["column"])) for r in all_mapping if resolved_map.get(r["column"]) in df.columns]
        raw_answers  = {mcol: clean(resp.get(dfcol, "")) for (mcol, dfcol) in needed_pairs}

        # warm embedding cache (unique answers in this row)
        uniq_texts = list({t for t in raw_answers.values() if t})
        for t in uniq_texts: _ = embed_cached(t)

        out = {}
        # ID
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
            mcol, qid, attr, qhint = r["column"], r["question_id"], r["attribute"], r.get("prompt_hint","")
            dfcol = resolved_map.get(mcol)
            if not dfcol: 
                continue  # skip unmapped
            ans = raw_answers.get(mcol, "")
            ai_flags.append(looks_ai_like(ans))
            vec = embed_cached(ans)

            # resolve qkey: question_id → else fuzzy by prompt
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
            if "_Q" in qid:
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

        # fill missing cells to keep structure
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # per-attribute averages & ranks + overall
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
        out["Overall Rank"] = next((label for label, lo, hi in OVERALL_BANDS if lo <= overall_total <= hi), "")
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

st.markdown("---")
st.caption("Resolves mapping columns to Kobo fields (normalize + fuzzy), then scores via SBERT centroids for each question, with attribute & global fallbacks.")
