import streamlit as st
import pandas as pd
import numpy as np
import json, unicodedata, re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
import io
import datetime as dt
import openpyxl

# =========================================
# CONSTANTS / CONFIG
# =========================================

EXEMPLARS_PATH = "DATASETS/advisory_exemplars_smart.cleaned.jsonl"
MAPPING_PATH   = "DATASETS/mapping.csv"

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

AI_PATTERNS = [
    r"\bas an ai\b", r"\bi am an ai\b", r"\bthis response (?:was|is) generated\b",
    r"[—–-]{2,}", r"\bdelve\b|\bholistic\b|\bparadigm\b|\bgarner\b"
]
AI_REGEX = re.compile("|".join(AI_PATTERNS), re.I)
FUZZY_THRESHOLD = 78
MIN_QA_OVERLAP = 0.05

# =========================================
# HELPERS
# =========================================

def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def looks_ai_like(t):
    t = clean(t)
    if not t: return False
    if AI_REGEX.search(t): return True
    sents = re.split(r"(?<=[.!?])\s+", t)
    starts = [s[:20].lower() for s in sents if s]
    return len(starts) >= 3 and len(set(starts)) <= max(1, len(starts)//3)

def read_jsonl(path):
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def cos_sim(a,b):
    if a is None or b is None: return -1e9
    return float(np.dot(a,b))

def qa_overlap(ans, qtext):
    at=set(re.findall(r"\w+", (ans or "").lower()))
    qt=set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt)/ (len(qt)+1.0)) if qt else 1.0

# =========================================
# LOAD STATIC DATA
# =========================================

@st.cache_resource
def load_mapping():
    m = pd.read_csv(MAPPING_PATH)
    m.columns = [c.lower().strip() for c in m.columns]
    if "prompt_hint" not in m.columns:
        m["prompt_hint"] = ""
    return [r for r in m.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

@st.cache_resource
def load_exemplars_and_centroids():
    exemplars = read_jsonl(EXEMPLARS_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

    def centroid(texts):
        if not texts: return None
        embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.mean(axis=0)

    def build_centroids_from_pairs(txts, scores):
        buckets = {0:[],1:[],2:[],3:[]}
        for t,s in zip(txts, scores):
            if t: buckets[int(s)].append(t)
        return {sc: centroid(batch) for sc,batch in buckets.items()}

    q_centroids = {k: build_centroids_from_pairs(v["texts"], v["scores"]) for k,v in by_qkey.items()}
    attr_centroids = {attr:{sc: centroid(txts) for sc,txts in bucks.items()} for attr, bucks in by_attr.items()}
    global_buckets={0:[],1:[],2:[],3:[]}
    for e in exemplars:
        sc=int(e.get("score",0)); txt=clean(e.get("text",""))
        if txt: global_buckets[sc].append(txt)
    global_centroids = {sc: centroid(txts) for sc,txts in global_buckets.items()}

    return embedder, by_qkey, q_centroids, attr_centroids, global_centroids, question_texts

def resolve_qkey(question_id, hint, q_centroids, by_qkey, question_texts):
    qid = (question_id or "").strip()
    if qid and qid in q_centroids:
        return qid
    hint = clean(hint or "")
    if not hint or not question_texts:
        return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted:
                return k
    return None

# =========================================
# SCORE FUNCTION (EXACT COLAB LOGIC)
# =========================================

def score_dataframe(df):
    mapping = load_mapping()
    embedder, by_qkey, q_centroids, attr_centroids, global_centroids, question_texts = load_exemplars_and_centroids()

    # Detect ID/dates/staff/duration
    date_pref=["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_pref if c in df.columns), None)
    dt_series = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.Series([pd.NaT]*len(df))

    staff_id_col = next((c for c in df.columns if c.lower()=="staff id"), None)

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end","_submission_time","SubmissionDate","submissiondate"] if c in df.columns), None)

    def parse_dt(s): 
        try: return pd.to_datetime(s,errors="coerce")
        except: return pd.NaT

    if start_col or end_col:
        start_dt=parse_dt(df[start_col]) if start_col else pd.NaT
        end_dt=parse_dt(df[end_col]) if end_col else pd.NaT
        duration_min=((end_dt-start_dt).dt.total_seconds()/60.0).round(2)
    else:
        duration_min=pd.Series([np.nan]*len(df))

    rows_out=[]

    for i,resp in df.iterrows():
        needed_cols=[r["column"] for r in mapping if r["column"] in df.columns]
        raw={c: clean(resp.get(c,"")) for c in needed_cols}
        uniq=list({t for t in raw.values() if t})
        if uniq:
            vecs=embedder.encode(uniq, convert_to_numpy=True, normalize_embeddings=True)
            vec_map=dict(zip(uniq, vecs))
        else:
            vec_map={}

        def embed_ans(t): return vec_map.get(clean(t),None)

        out={}
        if pd.notna(dt_series.iloc[i]):
            out["ID"]=pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["ID"]=str(i)
        out["Staff ID"]=str(resp.get(staff_id_col)) if staff_id_col else ""
        out["Duration_min"]=float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr={}
        ai_flags=[]

        for r in mapping:
            col, qid, attr, hint = r["column"], r["question_id"], r["attribute"], r.get("prompt_hint","")
            qn = (re.search(r"_Q(\d+)$", qid).group(1) if "_Q" in qid else qid)
            score_key=f"{attr}_Qn{qn}"
            rubric_key=f"{attr}_Rubric_Qn{qn}"

            ans=raw.get(col,"")
            ai_flags.append(looks_ai_like(ans))
            vec=embed_ans(ans)

            qkey=resolve_qkey(qid, hint, q_centroids, by_qkey, question_texts)
            sc=None

            if vec is not None:
                # q centroid
                if qkey and qkey in q_centroids:
                    sims={s: cos_sim(vec,v) for s,v in q_centroids[qkey].items() if v is not None}
                    if sims:
                        sc=max(sims,key=sims.get)
                        qtext=by_qkey[qkey]["question_text"]
                        if qa_overlap(ans,qtext or hint)<MIN_QA_OVERLAP:
                            sc=min(sc,1)
                # attr
                if sc is None and attr in attr_centroids:
                    sims={s:cos_sim(vec,v) for s,v in attr_centroids[attr].items() if v is not None}
                    if sims:
                        sc=max(sims,key=sims.get)
                        if qa_overlap(ans,hint)<MIN_QA_OVERLAP:
                            sc=min(sc,1)
                # global
                if sc is None:
                    sims={s:cos_sim(vec,v) for s,v in global_centroids.items() if v is not None}
                    if sims:
                        sc=max(sims,key=sims.get)
                        if qa_overlap(ans,hint)<MIN_QA_OVERLAP:
                            sc=min(sc,1)

            if sc is None:
                out[score_key]=""
                out[rubric_key]=""
            else:
                out[score_key]=int(sc)
                out[rubric_key]=BANDS[int(sc)]
                per_attr.setdefault(attr,[]).append(int(sc))

        total=0
        for attr in ORDERED_ATTRS:
            scores=per_attr.get(attr,[])
            if not scores:
                out[f"{attr}_Avg (0–3)"]=""
                out[f"{attr}_RANK"]=""
            else:
                avg=float(np.mean(scores))
                band=int(round(avg))
                total+=band
                out[f"{attr}_Avg (0–3)"]=round(avg,2)
                out[f"{attr}_RANK"]=BANDS[band]

        out["Overall Total (0–24)"]=total
        out["Overall Rank"]=next((label for label,lo,hi in OVERALL_BANDS if lo<=total<=hi),"")
        out["AI_suspected"]=bool(any(ai_flags))
        rows_out.append(out)

    res=pd.DataFrame(rows_out)

    # order cols
    def order_cols(cols):
        ordered=["ID","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            qns=sorted(set(int(re.search(r"_Qn(\d+)$",c).group(1)) for c in cols if c.startswith(attr+"_Qn") and re.search(r"_Qn(\d+)$",c)))
            for qn in qns:
                ordered += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI_suspected"]
        extras=[c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols]+extras

    res = res.reindex(columns=order_cols(res.columns))
    return res

# =========================================
# STREAMLIT UI
# =========================================

st.title("Advisory Skills Scoring — Offline Client Version")
file = st.file_uploader("Upload Kobo XLSX/CSV", type=["xlsx","csv"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        xls = pd.ExcelFile(file)
        sheet = next((s for s in xls.sheet_names if s.lower() in
                      ("data","survey","export","form","submissions","sheet1")), xls.sheet_names[0])
        df = pd.read_excel(file, sheet_name=sheet)

    st.success("✅ File loaded")
    if st.button("Score Advisory Responses"):
        with st.spinner("Scoring…"):
            res = score_dataframe(df)
        st.success("✅ Scoring complete")

        # download
        buf = io.BytesIO()
        res.to_excel(buf, index=False)
        st.download_button("Download results", buf.getvalue(),
                           file_name="advisory_scoring_output.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.dataframe(res.head())
