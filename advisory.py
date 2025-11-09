# advisory.py — Kobo → TF-IDF+SVD centroids (offline) → AI detection → Excel/Sheets
# No HuggingFace; No LLM; Fully offline scoring & AI-suspect signals.

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

import json, re, unicodedata, math, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests

from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

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

FUZZY_THRESHOLD     = 80
MIN_QA_OVERLAP      = 0.05

# ---- AI detection tuning ----
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.62))
NEAR_DUP_SIM         = float(st.secrets.get("NEAR_DUP_SIM", 0.92))
MIN_ANS_LEN_FOR_AI   = int(st.secrets.get("MIN_ANS_LEN_FOR_AI", 60))
MAX_SENT_AVG_LEN     = float(st.secrets.get("MAX_SENT_AVG_LEN", 40.0))
MIN_TTR              = float(st.secrets.get("MIN_TTR", 0.35))
MAX_TRANSITION_RATE  = float(st.secrets.get("MAX_TRANSITION_RATE", 0.4))

AI_TEMPLATE_PHRASES = [
    "as an ai","as a language model","i do not have access","i cannot access","based on the provided",
    "it is important to note","moving forward,","furthermore,","moreover,","in conclusion,",
    "this response","holistic approach","key takeaways","robust framework",
    "leveraging","best practices include","i recommend","i suggest",
    "from a strategic perspective","evidence-based","data-driven","mitigation strategies",
]
AI_MARKDOWN_CUES = ["```","**","__","###","- ","* ","— ","––","___","…","----"]

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

def qa_overlap(ans: str, qtext: str) -> float:
    at = set(re.findall(r"\w+", (ans or "").lower()))
    qt = set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

def show_status(ok: bool, msg: str) -> None:
    (st.success if ok else st.error)(msg)

# ==============================
# LOADERS
# ==============================
def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), "mapping must have: column, question_id, attribute"
    if "prompt_hint" not in m.columns: m["prompt_hint"] = ""
    m = m[m["attribute"].isin(ORDERED_ATTRS)].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists(): raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
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
            if r.status_code == 404: continue
            r.raise_for_status()
            payload = r.json()
            results = payload if isinstance(payload, list) else payload.get("results", [])
            if not results and "results" not in payload: results = payload
            df = pd.DataFrame(results)
            if not df.empty: df.columns = [str(c).strip() for c in df.columns]
            return df
        except requests.HTTPError:
            if r.status_code in (401, 403):
                st.error("Kobo auth failed: check KOBO_TOKEN and tenant.")
                return pd.DataFrame()
            if r.status_code == 404: continue
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()
    st.error("Could not fetch data. Check KOBO_BASE, KOBO_ASSET_ID, token permissions.")
    return pd.DataFrame()

# ==============================
# QUESTION_ID → KOBO COLUMN RESOLVER
# ==============================
QID_PREFIX_TO_SECTION = {"SAT":"A1","CT":"A2","ECI":"A3","CSF":"A4","FCP":"A5","ERI":"A6","SOA":"A7","CSE":"A8"}
QNUM_RX = re.compile(r"_Q(\d+)$")

def build_kobo_base_from_qid(question_id: str) -> str | None:
    if not question_id: return None
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m: return None
    qn = m.group(1); prefix = qid.split("_Q")[0]
    if prefix not in QID_PREFIX_TO_SECTION: return None
    section = QID_PREFIX_TO_SECTION[prefix]
    return f"Advisory/{section}_Section/{section}_{qn}"

def expand_possible_kobo_columns(base: str) -> list[str]:
    if not base: return []
    return [base, f"{base} :: Answer (text)", f"{base} :: English (en)", f"{base} - English (en)", f"{base}_labels", f"{base}_label"]

def _score_kobo_header(col: str, token: str) -> int:
    c = col.lower(); t = token.lower()
    if c == t: return 100
    s = 0
    if c.endswith("/"+t): s = max(s,95)
    if f"/{t}/" in c: s = max(s,92)
    if f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c: s = max(s,90)
    if t in c: s = max(s,80)
    if "english" in c or "label" in c: s += 3
    if "answer (text)" in c or "answer_text" in c or "text" in c: s += 2
    if "advisory/" in c or "/a" in c: s += 1
    return s

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    base = build_kobo_base_from_qid(question_id)
    token = None
    if question_id:
        qid = question_id.strip().upper()
        m = QNUM_RX.search(qid)
        if m:
            qn = m.group(1); prefix = qid.split("_Q")[0]; sect = QID_PREFIX_TO_SECTION.get(prefix)
            if sect: token = f"{sect}_{qn}"
    if base and base in df_cols: return base
    if base:
        for v in expand_possible_kobo_columns(base):
            if v in df_cols: return v
        for c in df_cols:
            if c.startswith(base): return c
    if token:
        best, bs = None, 0
        for col in df_cols:
            sc = _score_kobo_header(col, token)
            if sc > bs: bs, best = sc, col
        if best and bs >= 82: return best
    hint = clean(prompt_hint or "")
    if hint:
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=5)
        for col, score, _ in hits:
            if score >= 88: return col
    return None

# ==============================
# OFFLINE EMBEDDINGS: TF-IDF + SVD (LSA)
# ==============================
@st.cache_resource(show_spinner=False)
def build_text_encoder(corpus: list[str], n_components: int = 256):
    """Fit a TF-IDF+SVD encoder on corpus; return (vectorize_fn)."""
    corpus = [clean(t) for t in corpus if t and t.strip()]
    if not corpus:
        # fallback tiny corpus avoids SVD errors
        corpus = ["placeholder"]
    tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        ngram_range=(1,2), min_df=1, max_df=0.98, max_features=100_000
    )
    X = tfidf.fit_transform(corpus)
    k = min(n_components, min(X.shape)-1) if min(X.shape) > 1 else 1
    svd = TruncatedSVD(n_components=k, random_state=0)
    X_lsa = svd.fit_transform(X)
    norm = Normalizer(copy=False)
    X_lsa = norm.fit_transform(X_lsa)

    def encode(texts: list[str]) -> np.ndarray:
        if not texts: return np.zeros((0, k), dtype=np.float32)
        Xt = tfidf.transform([clean(t) for t in texts])
        Z = svd.transform(Xt)
        Z = norm.transform(Z)
        return Z.astype(np.float32)

    return encode

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return -1e9
    return float(np.dot(a, b))

# ==============================
# CENTROIDS (uses offline encoder)
# ==============================
def build_centroids(exemplars: list[dict], encode):
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
            by_qkey[key] = {"attribute": attr, "question_text": qtext, "by_score": {0:[],1:[],2:[],3:[]}}
            if qtext: question_texts.append(qtext)
        by_qkey[key]["by_score"][score].append(text)
        by_attr.setdefault(attr, {0:[],1:[],2:[],3:[]})
        by_attr[attr][score].append(text)

    def centroid(texts):
        texts = [t for t in texts if t]
        if not texts: return None
        M = encode(texts)
        if M.size == 0: return None
        c = M.mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-9)
        return c

    q_centroids = {k:{sc:centroid(txts) for sc,txts in pack["by_score"].items()} for k,pack in by_qkey.items()}
    attr_centroids = {attr:{sc:centroid(txts) for sc,txts in bucks.items()} for attr,bucks in by_attr.items()}

    global_buckets = {0:[],1:[],2:[],3:[]}
    for e in exemplars:
        sc = int(e.get("score",0)); txt = clean(e.get("text",""))
        if txt: global_buckets[sc].append(txt)
    global_centroids = {sc:centroid(txts) for sc,txts in global_buckets.items()}

    return q_centroids, attr_centroids, global_centroids, by_qkey, question_texts

def resolve_qkey(q_centroids, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in q_centroids: return qid
    hint = clean(prompt_hint or "")
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio) if (hint and question_texts) else None
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack["question_text"]) == wanted: return k
    return None

# ==============================
# AI DETECTION (heuristics + duplicates)
# ==============================
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RX    = re.compile(r"[A-Za-z']+")

def norm_text(t: str) -> str:
    t = clean(t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def trigram_set(t: str) -> set:
    tokens = norm_text(t).split()
    return set(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 3 else set()

def type_token_ratio(t: str) -> float:
    toks = WORD_RX.findall(t.lower())
    if not toks: return 0.0
    return len(set(toks)) / len(toks)

def transition_rate(sentences: list[str]) -> float:
    if not sentences: return 0.0
    transitions = {"furthermore","moreover","additionally","however","therefore","thus","meanwhile","consequently","importantly",
                   "in conclusion","in summary","to conclude","overall","first","second","third"}
    hits = 0
    for s in sentences:
        s2 = s.strip().lower()
        if any(s2.startswith(x) for x in transitions) or re.match(r"^\(?\d+[\).\s-]", s2):
            hits += 1
    return hits / len(sentences)

def markdown_cues_ratio(t: str) -> float:
    if not t: return 0.0
    total = sum(t.count(c) for c in AI_MARKDOWN_CUES)
    return min(1.0, total / max(1, len(t)//80))

def template_phrase_hits(t: str) -> int:
    tl = t.lower()
    return sum(1 for p in AI_TEMPLATE_PHRASES if p in tl)

def repetition_score(t: str) -> float:
    toks = norm_text(t).split()
    if len(toks) < 12: return 0.0
    from collections import Counter
    c = Counter(toks)
    top2 = [n for (_, n) in c.most_common(2)]
    rep = (sum(top2)/len(toks)) if top2 else 0.0
    return min(1.0, rep)

def ai_signal_score(answer: str, question_hint: str) -> tuple[float, list[str]]:
    flags = []
    a = clean(answer)
    if len(a) < MIN_ANS_LEN_FOR_AI:
        return 0.0, flags
    sents = [s for s in SENT_SPLIT.split(a) if s.strip()]
    avg_sent_len = (sum(len(WORD_RX.findall(s)) for s in sents)/max(1,len(sents))) if sents else 0.0
    ttr = type_token_ratio(a)
    trans = transition_rate(sents)
    md = markdown_cues_ratio(a)
    tpl = template_phrase_hits(a)
    rep = repetition_score(a)
    ov = qa_overlap(a, question_hint or "")

    features = {
        "avg_sent_len": min(1.0, max(0.0, (avg_sent_len - 20.0) / (MAX_SENT_AVG_LEN - 20.0))),
        "low_ttr":      1.0 - max(MIN_TTR, min(1.0, ttr)) if ttr < 1.0 else 0.0,
        "transitions":  min(1.0, trans / MAX_TRANSITION_RATE),
        "markdown":     md,
        "templates":    min(1.0, tpl / 3.0),
        "repetition":   rep,
        "low_overlap":  (1.0 - ov) if len(a) > 120 else 0.0,
    }
    w = {"avg_sent_len":0.14,"low_ttr":0.16,"transitions":0.12,"markdown":0.08,"templates":0.22,"repetition":0.16,"low_overlap":0.12}
    score = sum(features[k]*w[k] for k in w)

    if tpl >= 1: flags.append("template_phrases")
    if md > 0: flags.append("markdown/formatting")
    if trans > MAX_TRANSITION_RATE: flags.append("structured_transitions")
    if avg_sent_len > MAX_SENT_AVG_LEN: flags.append("very_long_sentences")
    if ttr < MIN_TTR: flags.append("low_lexical_diversity")
    if rep > 0.18: flags.append("repetition")
    if ov < 0.08: flags.append("low_QA_overlap")

    return float(max(0.0, min(1.0, score))), flags

# ==============================
# SCORING (centroids via LSA) + AI DETECTION
# ==============================
def build_encoder_from_exemplars_and_answers(exemplars, answers_pool):
    # Fit encoder on exemplars + answers (better coverage) — offline
    corpus = [clean(e.get("text","")) for e in exemplars if clean(e.get("text",""))]
    corpus += [t for t in answers_pool if t]
    return build_text_encoder(corpus, n_components=256)

def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    encode, q_centroids, attr_centroids, global_centroids,
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

    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

    # Resolve Kobo columns
    resolved_for_qid, missing_map_rows = {}, []
    for r in all_mapping:
        qid   = r["question_id"]; qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit: resolved_for_qid[qid] = hit
        else:   missing_map_rows.append((qid, qhint))

    with st.expander("🧭 Mapping → Kobo column resolution", expanded=False):
        if resolved_for_qid:
            st.dataframe(pd.DataFrame(list(resolved_for_qid.items())[:60], columns=["question_id","kobo_column"]))
        if missing_map_rows:
            st.warning(f"{len(missing_map_rows)} question_ids not found (up to 30 shown).")
            st.dataframe(pd.DataFrame(missing_map_rows[:30], columns=["question_id","prompt_hint"]))

    # Collect answers (for duplicates and vectorization)
    per_row_answers = []
    distinct_texts = set()
    for _, resp in df.iterrows():
        arow = {}
        for r in all_mapping:
            qid = r["question_id"]; col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                txt = clean(resp.get(col, ""))
                if txt: arow[qid] = txt; distinct_texts.add(txt)
        per_row_answers.append(arow)

    # Encode all distinct answers once
    if distinct_texts:
        A = encode(list(distinct_texts))
        # map text -> vector
        vec_map = {t: A[i] for i,t in enumerate(list(distinct_texts))}
    else:
        vec_map = {}

    # Duplicate/near-duplicate clusters (trigram Jaccard, bucketed)
    def norm_text(t: str) -> str:
        t = clean(t).lower()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    def trigram_set(t: str) -> set:
        tokens = norm_text(t).split()
        return set(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 3 else set()
    trigram_index = {t: trigram_set(t) for t in distinct_texts}
    def sig32(t: str) -> int:
        return int(hashlib.md5(norm_text(t).encode("utf-8")).hexdigest()[:8], 16)
    buckets = {}
    for t in distinct_texts:
        buckets.setdefault(sig32(t), []).append(t)
    near_dup_map = {}
    for _, group in buckets.items():
        if len(group) == 1:
            near_dup_map[group[0]] = (1, 0.0); continue
        sets = {t: trigram_index[t] for t in group}
        for t in group:
            s = sets[t]
            if not s: near_dup_map[t] = (len(group), 0.0); continue
            sims = []
            for u in group:
                if u == t: continue
                su = sets[u]
                if not su: continue
                inter = len(s & su); union = len(s | su) or 1
                jacc = inter/union
                if jacc >= NEAR_DUP_SIM: sims.append(jacc)
            near_dup_map[t] = (max(1, 1+len(sims)), float(len(sims))/max(1, len(group)-1))

    # ---------- Scoring ----------
    rows_out = []
    for i, resp in df.iterrows():
        out = {}
        out["ID"] = (pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
                     if pd.notna(dt_series.iloc[i]) else str(i))
        staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id","staff_id")), None)
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr = {}
        ai_scores_local = []
        ai_flags_local  = []

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            dfcol = resolved_for_qid.get(qid)
            if not dfcol or dfcol not in df.columns: continue
            ans = clean(resp.get(dfcol, "")); 
            if not ans: continue

            # AI detection per answer
            ai_score, ai_flags = ai_signal_score(ans, qhint)
            ai_scores_local.append(ai_score)
            qn_str = qid.split("_Q")[-1] if "_Q" in (qid or "") else "?"
            ai_flags_local.extend([f"{attr}_Q{qn_str}:{flag}" for flag in ai_flags])

            # vector
            vec = vec_map.get(ans)
            # only Q1..Q4
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4): 
                continue

            # centroid band
            sc = None
            if vec is not None:
                # question-level
                sims = {}
                qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)
                if qkey and qkey in q_centroids:
                    sims = {s: cos_sim(vec, v) for s, v in q_centroids[qkey].items() if v is not None}
                    if sims:
                        sc = max(sims, key=sims.get)
                        qtext = (by_qkey.get(qkey, {}) or {}).get("question_text","")
                        if qa_overlap(ans, qtext or qhint) < MIN_QA_OVERLAP:
                            sc = min(sc, 1)
                # attribute-level
                if sc is None and attr in attr_centroids:
                    sims = {s: cos_sim(vec, v) for s, v in attr_centroids[attr].items() if v is not None}
                    if sims:
                        sc = max(sims, key=sims.get)
                        if qa_overlap(ans, qhint) < MIN_QA_OVERLAP:
                            sc = min(sc, 1)
                # global-level
                if sc is None:
                    sims = {s: cos_sim(vec, v) for s, v in global_centroids.items() if v is not None}
                    if sims:
                        sc = max(sims, key=sims.get)
                        if qa_overlap(ans, qhint) < MIN_QA_OVERLAP:
                            sc = min(sc, 1)

            # write per-question columns (score + AI signals)
            sk = f"{attr}_Qn{qn}"
            rk = f"{attr}_Rubric_Qn{qn}"
            ak = f"{attr}_AI_score_Qn{qn}"
            af = f"{attr}_AI_flags_Qn{qn}"

            if sc is None:
                out.setdefault(sk, ""); out.setdefault(rk, "")
            else:
                out[sk]  = int(sc)
                out[rk]  = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

            out[ak] = round(float(ai_score), 3)
            out[af] = ";".join(sorted(set(ai_flags))) if ai_flags else ""

        # Defaults for missing cells
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")
                out.setdefault(f"{attr}_AI_score_Qn{qn}", "")
                out.setdefault(f"{attr}_AI_flags_Qn{qn}", "")

        # Attribute avgs + overall
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

        # Row-level AI aggregation
        ai_scores_local = [s for s in ai_scores_local if isinstance(s,(int,float))]
        out["AI_score_max"]  = round(max(ai_scores_local) if ai_scores_local else 0.0, 3)
        out["AI_score_avg"]  = round(float(np.mean(ai_scores_local)) if ai_scores_local else 0.0, 3)
        out["AI_flags_concat"] = ";".join(sorted(set(ai_flags_local))) if ai_flags_local else ""

        # Duplicate cluster metrics
        dup_sizes, dup_pcts = [], []
        for txt in per_row_answers[i].values():
            if not txt: continue
            size, pct = near_dup_map.get(txt, (1,0.0))
            dup_sizes.append(size); dup_pcts.append(pct)
        out["DupClusterSize"] = max(dup_sizes) if dup_sizes else 1
        out["NearDupPct"]     = round(max(dup_pcts), 3) if dup_pcts else 0.0

        out["AI_suspected"] = bool(out["AI_score_max"] >= AI_SUSPECT_THRESHOLD or out["DupClusterSize"] >= 3 or out["NearDupPct"] >= 0.5)
        rows_out.append(out)

    res_df = pd.DataFrame(rows_out)

    def order_cols(cols):
        ordered = ["ID","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [
                    f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}",
                    f"{attr}_AI_score_Qn{qn}", f"{attr}_AI_flags_Qn{qn}",
                ]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += [
            "Overall Total (0–24)", "Overall Rank",
            "DupClusterSize", "NearDupPct",
            "AI_score_max", "AI_score_avg", "AI_flags_concat", "AI_suspected"
        ]
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

# ==============================
# Google Sheets
# ==============================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME", "Advisory")

def _normalize_sa_dict(raw: dict) -> dict:
    if not raw: raise ValueError("gcp_service_account missing in secrets.")
    sa = dict(raw)
    if "token_ur" in sa and "token_uri" not in sa: sa["token_uri"] = sa.pop("token_ur")
    if sa.get("private_key") and "\\n" in sa["private_key"]: sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    sa.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    sa.setdefault("auth_uri", "https://accounts.google.com/o/oauth2/auth")
    sa.setdefault("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs")
    required = ["type","project_id","private_key_id","private_key","client_email","client_id","token_uri"]
    missing = [k for k in required if not sa.get(k)]
    if missing: raise ValueError(f"gcp_service_account missing fields: {', '.join(missing)}")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets.get("gcp_service_account"))
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

def _to_a1_col(n: int) -> str:
    s = []
    while n > 0:
        n, r = divmod(n - 1, 26)
        s.append(chr(65 + r))
        n //= 26
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

def _open_ws_by_key() -> gspread.Worksheet:
    key = st.secrets.get("GSHEETS_SPREADSHEET_KEY")
    ws_name = DEFAULT_WS_NAME
    if not key: raise ValueError("GSHEETS_SPREADSHEET_KEY not set in st.secrets.")
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
# STAR SCHEMA
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
        try: qn = int(c.split("_Qn")[1])
        except: continue
        rubric_col = f"{attr}_Rubric_Qn{qn}"
        r = scored[["ID","Staff ID"]].copy()
        r["Attribute"] = attr; r["QuestionNo"] = qn
        r["Score"] = scored[c]; r["RubricBand"] = scored[rubric_col] if rubric_col in scored.columns else np.nan
        qrows.append(r)
    fact_question = pd.concat(qrows, ignore_index=True) if qrows else pd.DataFrame(
        columns=["ID","Staff ID","Attribute","QuestionNo","Score","RubricBand"]
    )

    arows = []
    for attr in attributes:
        avg_col = f"{attr}_Avg (0–3)"; rank_col = f"{attr}_RANK"
        r = scored[["ID","Staff ID"]].copy()
        r["Attribute"] = attr
        r["AvgScore"] = scored.get(avg_col); r["RankBand"] = scored.get(rank_col)
        arows.append(r)
    fact_attribute = pd.concat(arows, ignore_index=True) if arows else pd.DataFrame(
        columns=["ID","Staff ID","Attribute","AvgScore","RankBand"]
    )

    sub_cols = ["ID","Staff ID","Duration_min","Overall Total (0–24)","Overall Rank",
                "DupClusterSize","NearDupPct","AI_score_max","AI_score_avg","AI_suspected"]
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
        dim_date["year"] = dt.dt.year; dim_date["quarter"] = dt.dt.quarter
        dim_date["month"] = dt.dt.month; dim_date["day"] = dt.dt.day
        dim_date["week"] = dt.dt.isocalendar().week.astype(int)
        dim_date["dow"] = dt.dt.dayofweek
        dim_date["month_name"] = dt.dt.month_name(); dim_date["dow_name"] = dt.dt.day_name()

    dim_staff = submission[["Staff ID"]].rename(columns={"Staff ID":"staff_natural_key"}).drop_duplicates()
    dim_staff["staff_key"] = dim_staff["staff_natural_key"].astype("category").cat.codes + 1
    dim_staff = dim_staff[["staff_key","staff_natural_key"]]

    dim_attribute = pd.DataFrame({"attribute_name": attributes})
    if not dim_attribute.empty:
        dim_attribute["attribute_key"] = dim_attribute["attribute_name"].astype("category").cat.codes + 1
        dim_attribute = dim_attribute[["attribute_key","attribute_name"]]

    staff_map = dict(zip(dim_staff["staff_natural_key"], dim_staff["staff_key"]))
    attr_map  = dict(zip(dim_attribute["attribute_name"], dim_attribute["attribute_key"]))

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
                                 "Overall Total (0–24)","Overall Rank",
                                 "DupClusterSize","NearDupPct","AI_score_max","AI_score_avg","AI_suspected"]]
    return {"fact_attribute":fact_attribute,"fact_question":fact_question,
            "dim_staff":dim_staff,"dim_attribute":dim_attribute,"dim_date":dim_date,"submission":submission_out}

# ==============================
# UI
# ==============================
def main():
    st.title("📊 Advisory Scoring + AI Detection (Offline — no HuggingFace)")
    st.caption(f"AI_SUSPECT_THRESHOLD={AI_SUSPECT_THRESHOLD} · NEAR_DUP_SIM={NEAR_DUP_SIM}")

    # Load mapping/exemplars
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

    # Build encoder on exemplars + (later) answers
    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found.")
        st.stop()

    # Pool answers to include vocabulary in encoder
    answers_pool = []
    df_cols = list(df.columns)
    tmp_map = mapping.to_dict(orient="records")
    resolved = {}
    for r in tmp_map:
        qid = r["question_id"]; hit = resolve_kobo_column_for_mapping(df_cols, qid, r.get("prompt_hint",""))
        if hit: resolved[qid] = hit
    for _, row in df.iterrows():
        for r in tmp_map:
            col = resolved.get(r["question_id"])
            if col and col in df.columns:
                t = clean(row.get(col, ""))
                if t: answers_pool.append(t)

    with st.spinner("Building offline text encoder (TF-IDF+SVD)..."):
        encode = build_encoder_from_exemplars_and_answers(exemplars, answers_pool)

    with st.spinner("Building centroids..."):
        q_c, a_c, g_c, by_q, qtexts = build_centroids(exemplars, encode)

    st.caption("Fetched sample:")
    st.dataframe(df.head(), use_container_width=True)

    with st.spinner("Scoring + AI detection..."):
        scored_df = score_dataframe(df, mapping, encode, q_c, a_c, g_c, by_q, qtexts)

    st.success("✅ Done.")
    st.dataframe(scored_df.head(40), use_container_width=True)

    st.download_button(
        "⬇️ Download Excel",
        data=to_excel_bytes(scored_df),
        file_name="Advisory_Scoring_AI_Offline.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    with st.expander("📤 Google Sheets export", expanded=True):
        st.write("Spreadsheet key:", st.secrets.get("GSHEETS_SPREADSHEET_KEY") or "⚠️ Not set")
        st.write("Worksheet name:", DEFAULT_WS_NAME)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload to Google Sheets", use_container_width=True):
                ok, msg = upload_df_to_gsheets(scored_df)
                show_status(ok, msg)
        with col2:
            st.caption("Tune AI_SUSPECT_THRESHOLD/NEAR_DUP_SIM in secrets for stricter or looser flags.")

if __name__ == "__main__":
    main()
