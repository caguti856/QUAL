# ============================================================
# 📈 Advisory Scoring Engine — Kobo → Excel / Power BI / Sheets
# Refactored + modular + cleaner architecture
# ============================================================

import streamlit as st
st.set_page_config(page_title="Advisory Scoring Dashboard", layout="wide")

# --------- Core Imports ----------
import re, json, unicodedata
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# --------- App Constants / Paths ----------
DATASETS_DIR = Path("DATASETS")
MAPPING_PATH = DATASETS_DIR / "mapping.csv"
EXEMPLARS_PATH = DATASETS_DIR / "advisory_exemplars_smart.cleaned.jsonl"

KOBO_BASE = st.secrets.get("KOBO_BASE", "")
KOBO_TOKEN = st.secrets.get("KOBO_TOKEN", "")
KOBO_ASSET_ID = st.secrets.get("KOBO_ASSET_ID", "")

ORDERED_ATTRS = [
    "Strategic & analytical thinking","Credibility & trustworthiness",
    "Effective communication & influence","Client & stakeholder focus",
    "Fostering collaboration & partnership","Ensuring relevance & impact",
    "Solution orientation & adaptability","Capacity strengthening & empowerment support",
]

BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 21, 24),
    ("Strategic Advisor",       16, 20),
    ("Emerging Advisor",        10, 15),
    ("Needs Capacity Support",   0,  9),
]

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

# ============================================================
# Helpers
# ============================================================
def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s).strip()

def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def cos_sim(a,b):
    return float(np.dot(a,b)) if a is not None and b is not None else -1e9

def qa_overlap(ans, q):
    a = set(re.findall(r"\w+", ans.lower()))
    b = set(re.findall(r"\w+", q.lower()))
    return (len(a&b)/(len(b)+1)) if b else 1.0

AI_FLAG = re.compile(r"(as an ai\b|i am an ai|---|____)", re.I)
def looks_ai_like(t): return bool(AI_FLAG.search(clean(t)))

# ============================================================
# Kobo
# ============================================================
def kobo_url(a, kind="submissions"):
    return f"{KOBO_BASE}/api/v2/assets/{a}/{kind}/?format=json"

@st.cache_data(ttl=300)
def fetch_kobo():
    if not KOBO_ASSET_ID or not KOBO_TOKEN:
        st.error("✅ Add KOBO_ASSET_ID & KOBO_TOKEN to secrets")
        return pd.DataFrame()

    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for k in ("submissions","data"):
        r = requests.get(kobo_url(KOBO_ASSET_ID,k), headers=headers, timeout=40)
        if r.status_code in (401,403):
            st.error("❌ Kobo auth failed")
            return pd.DataFrame()
        try:
            data = r.json()
            rows = data if isinstance(data,list) else data.get("results",[])
            return pd.DataFrame(rows)
        except:
            pass
    return pd.DataFrame()

# ============================================================
# Load mapping + exemplars
# ============================================================
def load_mapping():
    m = pd.read_csv(MAPPING_PATH)
    m.columns = [c.lower().strip() for c in m.columns]
    if "prompt_hint" not in m: m["prompt_hint"] = ""
    return m[m["attribute"].isin(ORDERED_ATTRS)]

def load_exemplars():
    with open(EXEMPLARS_PATH,"r",encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ============================================================
# Semantic model build
# ============================================================
@st.cache_resource
def build_centroids(exemplars):
    emb = embedder()

    attr_bins = {a:{i:[] for i in range(4)} for a in ORDERED_ATTRS}
    global_bins = {i:[] for i in range(4)}
    q_bins = {}

    for e in exemplars:
        q = clean(e["question_id"])
        t = clean(e["text"])
        a = clean(e["attribute"])
        s = int(e["score"])

        attr_bins[a][s].append(t)
        global_bins[s].append(t)
        q_bins.setdefault(q,{i:[] for i in range(4)})[s].append(t)

    def cent(texts):
        if not texts: return None
        v = emb.encode(texts, normalize_embeddings=True)
        return v.mean(axis=0)

    return (
        {q:{s:cent(txt) for s,txt in d.items()} for q,d in q_bins.items()},
        {a:{s:cent(txt) for s,txt in d.items()} for a,d in attr_bins.items()},
        {s:cent(txt) for s,txt in global_bins.items()}
    )

# ============================================================
# Scoring logic
# ============================================================
def embed_cached(text, cache):
    t = clean(text)
    if not t: return None
    if t not in cache:
        cache[t] = embedder().encode(t, normalize_embeddings=True)
    return cache[t]

def score_rows(df, mapping, qC, aC, gC):
    staff_col = next((c for c in df.columns if c.lower()=="staff id"), None)
    dt_col = next((c for c in df.columns if "submission" in c.lower() or c.lower()=="end"), None)

    results = []
    for _,r in df.iterrows():
        row = {"Staff ID": r.get(staff_col,""), "ID": clean(str(r.get(dt_col)))}
        per_attr = {}
        ai_flag = []

        cache = {}
        for _,m in mapping.iterrows():
            qid = m["question_id"]
            attr = m["attribute"]
            col = next((c for c in df.columns if qid.lower().split("_")[0] in c.lower()), None)
            ans = clean(r.get(col,""))

            if not ans: continue
            ai_flag.append(looks_ai_like(ans))
            v = embed_cached(ans, cache)

            # pick best level
            qS = qC.get(qid,{})
            sims = {s:cos_sim(v,cent) for s,cent in qS.items() if cent is not None}
            if sims: sc = max(sims,key=sims.get)
            else:
                sims = {s:cos_sim(v,cent) for s,cent in aC[attr].items() if cent is not None}
                sc = max(sims,key=sims.get) if sims else max(gC,key=lambda s:cos_sim(v,gC[s]))

            # penalize if off-topic
            if qa_overlap(ans, m["prompt_hint"]) < MIN_QA_OVERLAP: sc = min(sc,1)

            row[f"{attr}_{qid}"] = sc
            per_attr.setdefault(attr,[]).append(sc)

        tot = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr,[])
            if scores:
                avg = round(float(np.mean(scores)),2)
                band = int(round(avg))
                tot += band
                row[f"{attr}_Avg"] = avg
                row[f"{attr}_Rank"] = BANDS[band]
            else:
                row[f"{attr}_Avg"] = ""
                row[f"{attr}_Rank"] = ""

        row["Overall"] = tot
        row["Overall Rank"] = next((n for n,l,h in [(x,*y) for x,y in zip(OVERALL_BANDS,[i[1:] for i in OVERALL_BANDS])] if l<=tot<=h),"")
        row["AI_Flag"] = any(ai_flag)

        results.append(row)
    return pd.DataFrame(results)

# ============================================================
# UI
# ============================================================
st.title("📊 Advisory Scoring — Automated MEAL Engine")

if st.button("🚀 Run Scoring", use_container_width=True):
    st.info("Loading metadata and model...")
    mapping = load_mapping()
    exemplars = load_exemplars()
    qC,aC,gC = build_centroids(exemplars)

    st.info("Fetching Kobo submissions...")
    df = fetch_kobo()
    if df.empty: st.stop()

    st.success(f"Loaded {len(df)} records")

    st.info("Scoring…")
    scored = score_rows(df, mapping, qC,aC,gC)
    st.success("✅ Completed!")

    st.dataframe(scored.head())

    st.download_button("⬇️ Download Excel", scored.to_csv(index=False).encode(), "scores.csv")

    st.write("Done ✅ You can upload cleaned scores to Power BI or Sheets now.")
