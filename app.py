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
MIN_QA_OVERLAP  = 0.1

# ---------- helpers ----------
def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

AI_PATTERNS = [
    r"\bas an ai\b", r"\bi am an ai\b", r"\bthis response (?:was|is) generated\b",
    r"[—–-]{2,}", r"\bdelve\b|\bholistic\b|\bparadigm\b|\bgarner\b"
]
AI_REGEX = re.compile("|".join(AI_PATTERNS), re.I)
def looks_ai_like(t):
    t = clean(t)
    if not t: return False
    if AI_REGEX.search(t): return True
    sents = re.split(r"(?<=[.!?])\s+", t)
    starts = [s[:20].lower() for s in sents if s]
    return len(starts) >= 3 and len(set(starts)) <= max(1, len(starts)//3)

def load_mapping_from_path(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    for need in ("column","question_id","attribute"):
        assert need in df.columns, f"mapping must have '{need}'"
    if "prompt_hint" not in df.columns: df["prompt_hint"] = ""
    return df

# ==============================
# KOBO DATA FETCHER
# ==============================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID or not KOBO_TOKEN:
        return pd.DataFrame()

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
            return df
        except requests.HTTPError:
            if r.status_code in (401,403):
                st.error("Kobo auth failed: check KOBO_TOKEN / tenant.")
                return pd.DataFrame()
            if r.status_code == 404: continue
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_responses_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        xls = pd.ExcelFile(p)
        sheet = next((s for s in xls.sheet_names if s.lower() in
                      ("data","survey","export","form","submissions","sheet1")), xls.sheet_names[0])
        df = pd.read_excel(p, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ==============================
# SEMANTIC CENTROIDS
# ==============================
def centroid(texts, embedder):
    if not texts: return None
    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embs.mean(axis=0)

def build_centroids_from_pairs(texts, scores):
    buckets = {0:[],1:[],2:[],3:[]}
    for t,s in zip(texts,scores):
        if t: buckets[int(s)].append(t)
    return buckets

def build_centroids(exemplars):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
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
        by_attr.setdefault(attr,{0:[],1:[],2:[],3:[]})
        by_attr[attr][score].append(text)

    q_centroids = {k:{s: centroid(v, embedder) for s,v in build_centroids_from_pairs(v["texts"],v["scores"]).items()} for k,v in by_qkey.items()}
    attr_centroids = {attr:{sc: centroid(txts, embedder) for sc, txts in bucks.items()} for attr, bucks in by_attr.items()}
    global_buckets = {0:[],1:[],2:[],3:[]}
    for e in exemplars:
        sc = int(e.get("score",0)); txt = clean(e.get("text",""))
        if txt: global_buckets[sc].append(txt)
    global_centroids = {sc: centroid(txts, embedder) for sc, txts in global_buckets.items()}
    return q_centroids, attr_centroids, global_centroids, by_qkey, question_texts

def resolve_qkey(question_id, prompt_hint, q_centroids, by_qkey, question_texts):
    qid = (question_id or "").strip()
    if qid and qid in q_centroids: return qid
    hint = clean(prompt_hint or "")
    if not hint or not question_texts: return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted: return k
    return None

def cos_sim(a,b):
    if a is None or b is None: return -1e9
    return float(np.dot(a,b))

def qa_overlap(ans,qtext):
    at = set(re.findall(r"\w+", (ans or "").lower()))
    qt = set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

# ==============================
# SCORING
# ==============================
def score_dataframe(df, mapping, q_centroids, attr_centroids, global_centroids, by_qkey, question_texts):
    rows_out = []
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for i, resp in df.iterrows():
        needed_cols = [r["column"] for r in all_mapping if r["column"] in df.columns]
        raw_answers = {col: clean(resp.get(col,"")) for col in needed_cols}

        uniq_texts = list({t for t in raw_answers.values() if t})
        if uniq_texts:
            uniq_vecs = embedder.encode(uniq_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            vec_map = dict(zip(uniq_texts, uniq_vecs))
        else:
            vec_map = {}

        def embed_ans(text): t=clean(text); return vec_map.get(t,None)

        out = {}
        out["ID"] = str(resp.get("ID", resp.index))
        out["Staff ID"] = str(resp.get("Staff ID",""))
        out["Duration_min"] = float(resp.get("Duration_min",np.nan)) if "Duration_min" in resp else ""

        per_attr = {}
        ai_flags = []

        for r in all_mapping:
            col, qid, attr, qhint = r["column"], r["question_id"], r["attribute"], r.get("prompt_hint","")
            qn = (re.search(r"_Q(\d+)$", qid).group(1) if "_Q" in qid else qid)
            score_key  = f"{attr}_Qn{qn}"
            rubric_key = f"{attr}_Rubric_Qn{qn}"

            ans = raw_answers.get(col,"")
            ai_flags.append(looks_ai_like(ans))
            vec = embed_ans(ans)

            qkey = resolve_qkey(qid, qhint, q_centroids, by_qkey, question_texts)
            sc = None
            if vec is not None:
                if qkey and qkey in q_centroids:
                    sims = {s: cos_sim(vec,v) for s,v in q_centroids[qkey].items() if v is not None}
                    if sims:
                        sc = max(sims, key=sims.get)
                        qtext = by_qkey[qkey]["question_text"]
                        if qa_overlap(ans,qtext or qhint) < MIN_QA_OVERLAP:
                            sc = min(sc,1)
                if sc is None and attr in attr_centroids:
                    sims = {s: cos_sim(vec,v) for s,v in attr_centroids[attr].items() if v is not None}
                    if sims: sc = max(sims,key=sims.get); 
                    if qa_overlap(ans,qhint)<MIN_QA_OVERLAP: sc=min(sc,1)
                if sc is None:
                    sims = {s: cos_sim(vec,v) for s,v in global_centroids.items() if v is not None}
                    if sims: sc = max(sims,key=sims.get); 
                    if qa_overlap(ans,qhint)<MIN_QA_OVERLAP: sc=min(sc,1)

            if sc is None:
                out[score_key] = ""
                out[rubric_key] = ""
            else:
                out[score_key] = int(sc)
                out[rubric_key] = BANDS[int(sc)]
                per_attr.setdefault(attr,[]).append(int(sc))

        # per-attribute averages / overall
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr,[])
            if not scores:
                out[f"{attr}_Avg (0–3)"] = ""
                out[f"{attr}_RANK"] = ""
            else:
                avg = float(np.mean(scores)); band=int(round(avg))
                overall_total += band
                out[f"{attr}_Avg (0–3)"] = round(avg,2)
                out[f"{attr}_RANK"] = BANDS[band]

        out["Overall Total (0–24)"] = overall_total
        out["Overall Rank"] = next((label for label,lo,hi in OVERALL_BANDS if lo<=overall_total<=hi),"")
        out["AI_suspected"] = bool(any(ai_flags))
        rows_out.append(out)

    return pd.DataFrame(rows_out)

# ==============================
# EXPORTS
# ==============================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer: df.to_excel(writer,index=False)
    return bio.getvalue()

def push_to_powerbi(df: pd.DataFrame) -> tuple[bool,str]:
    if not POWERBI_PUSH_URL: return False,"POWERBI_PUSH_URL not set in secrets."
    send_df = df.copy()
    for c in send_df.columns:
        if send_df[c].dtype==object: send_df[c]=send_df[c].astype(str).str.slice(0,4000)
    try:
        r = requests.post(POWERBI_PUSH_URL,json=send_df.to_dict(orient="records"),timeout=60)
        if r.status_code in (200,202): return True,"Success"
        return False,f"{r.status_code} {r.text[:300]}"
    except Exception as e: return False,str(e)

# ==============================
# STREAMLIT UI
# ==============================
st.title("📊 Advisory Scoring: Kobo → Excel + Power BI")

if st.button("🚀 Fetch & Score All", type="primary", use_container_width=True):
    mapping = load_mapping_from_path(MAPPING_PATH)
    exemplars = read_jsonl(EXEMPLARS_PATH)
    q_centroids, attr_centroids, global_centroids, by_qkey, question_texts = build_centroids(exemplars)

    # load Kobo first, fallback local
    df = fetch_kobo_dataframe()
    if df.empty:
        st.info("Kobo empty or unavailable, loading local dataset...")
        RESPONSES_PATH = DATASETS_DIR / "responses.xlsx"
        df = load_responses_any(RESPONSES_PATH)
    if df.empty:
        st.warning("No submissions found.")
        st.stop()

    # normalize columns
    df.columns = [c.lower().replace("/","_").replace(" ","_") for c in df.columns]

    # optional: map Kobo columns to question_id
    # expand kobo_to_qid dict as needed
    kobo_to_qid = {row["column"].lower():row["question_id"] for idx,row in mapping.iterrows()}
    df.rename(columns=kobo_to_qid, inplace=True)

    # Staff ID & Duration
    if "staff_id" in df.columns: df["Staff ID"]=df["staff_id"].astype(str)
    else: df["Staff ID"]=""
    if "start" in df.columns and "end" in df.columns:
        df["Duration_min"]=((pd.to_datetime(df["end"],errors="coerce")-pd.to_datetime(df["start"],errors="coerce")).dt.total_seconds()/60).round(2)
    else: df["Duration_min"]=np.nan

    st.caption("Sample of responses:")
    st.dataframe(df.head(), use_container_width=True)

    with st.spinner("Scoring responses..."):
        scored_df = score_dataframe(df,mapping,q_centroids,attr_centroids,global_centroids,by_qkey,question_texts)
    st.success("✅ Scoring complete.")

    st.dataframe(scored_df.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(scored_df),
        file_name="Individual_Advisory_Scoring_Sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    ok,msg = push_to_powerbi(scored_df)
    if ok: st.success("📤 Pushed scored data to Power BI successfully.")
    else: st.error(f"❌ Failed to push to Power BI: {msg}")
