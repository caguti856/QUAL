# advisory.py — Kobo -> REQUIRED Hybrid (Centroids + Online LLM) -> Router -> Sheets
# Fast path: preprocessing first (mapping->answers->embeddings->centroids), disk caches, matrix sims, optional LLM skip/parallel.

import streamlit as st
import json, re, unicodedata, time, hashlib, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

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

CASE1_TEXT       = st.secrets.get("CASE1", "")  # case study text from secrets

# OpenAI-compatible LLM (REQUIRED unless centroid-only enabled)
LLM_API_BASE     = (st.secrets.get("LLM_API_BASE", "") or "").rstrip("/")
LLM_API_KEY      = st.secrets.get("LLM_API_KEY", "")
LLM_MODEL        = st.secrets.get("LLM_MODEL", "")
LLM_TEMPERATURE  = float(st.secrets.get("LLM_TEMPERATURE", 0.2))
LLM_TIMEOUT_SEC  = int(st.secrets.get("LLM_TIMEOUT_SEC", 60))

# Knobs
FUZZY_THRESHOLD  = 80
MIN_QA_OVERLAP   = 0.05
MIN_CONF_AUTO    = float(st.secrets.get("MIN_CONF_AUTO", 0.78))
MAX_DISAGREE     = int(st.secrets.get("MAX_DISAGREE", 1))

# Speed/Cost knobs
SKIP_LLM_IF_CONF  = float(st.secrets.get("SKIP_LLM_IF_CONF", 0.86))   # if centroid conf >= this AND overlap ok => skip LLM
SKIP_LLM_MIN_OVLP = float(st.secrets.get("SKIP_LLM_MIN_OVLP", 0.08))
MAX_PARALLEL_LLM  = int(st.secrets.get("MAX_PARALLEL_LLM", 6))
CENTROID_ONLY     = bool(st.secrets.get("CENTROID_ONLY", False))      # hard disable LLM (useful while routing is not ready)

# Google Sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME", "Advisory")

# Labels
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

# Caches
CACHE_DIR = Path(".st_cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
def _cache_path(name: str) -> Path: return CACHE_DIR / f"{name}.pkl"
def _save_pickle(p: Path, obj): 
    with p.open("wb") as f: pickle.dump(obj, f)
def _load_pickle(p: Path):
    try:
        with p.open("rb") as f: return pickle.load(f)
    except Exception:
        return None

# ==============================
# SMALL HELPERS
# ==============================
def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def qa_overlap(ans: str, qtext: str) -> float:
    at = set(re.findall(r"\w+", (ans or "").lower()))
    qt = set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

AI_RX = re.compile(r"(?:-{3,}|—{2,}|_{2,}|\.{4,}|as an ai\b|i am an ai\b)", re.I)
def looks_ai_like(t): return bool(AI_RX.search(clean(t)))

# ==============================
# LOADERS (mapping / exemplars)
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

# ==============================
# KOBO FETCH
# ==============================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe(sample_n: int | None = None) -> pd.DataFrame:
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
            if sample_n and len(df) > sample_n:
                df = df.sample(sample_n, random_state=42).reset_index(drop=True)
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
# QID → KOBO COLUMN RESOLVER
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
# EMBEDDINGS / CENTROIDS (CACHED)
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def _build_centroids_cached(exemplars: list[dict]):
    # cache key from exemplars length + simple hash
    raw_key = json.dumps([e.get("question_id","")+e.get("question_text","")+str(e.get("score","")) for e in exemplars])[:200000]
    key = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
    cp = _cache_path(f"centroids_{key}")
    cached = _load_pickle(cp)
    if cached: return cached

    by_qkey, by_attr, question_texts = {}, {}, []
    def C(x): return clean(x)
    for e in exemplars:
        qid, qtext = C(e.get("question_id","")), C(e.get("question_text",""))
        score, text = int(e.get("score",0)), C(e.get("text",""))
        attr = C(e.get("attribute",""))
        if not qid and not qtext: continue
        keyq = qid if qid else qtext
        if keyq not in by_qkey:
            by_qkey[keyq] = {"attribute": attr, "question_text": qtext, "scores": [], "texts": []}
            if qtext: question_texts.append(qtext)
        by_qkey[keyq]["scores"].append(score)
        by_qkey[keyq]["texts"].append(text)
        by_attr.setdefault(attr, {0:[],1:[],2:[],3:[]})
        by_attr[attr][score].append(text)

    embedder = get_embedder()

    def centroid(texts):
        if not texts: return None
        embs = embedder.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
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
        sc = int(e.get("score",0)); txt = C(e.get("text",""))
        if txt: global_buckets[sc].append(txt)
    global_centroids = {sc: centroid(txts) for sc, txts in global_buckets.items()}

    # also precompute band matrices
    def _as_matrix(centroid_map: dict[int,np.ndarray]):
        items = [(b,v) for b,v in centroid_map.items() if v is not None]
        if not items:
            return [], None
        items.sort(key=lambda x:x[0])
        bands = [b for b,_ in items]
        mat = np.vstack([v for _,v in items])  # normalized
        return bands, mat

    attr_mats = {attr: _as_matrix(m) for attr, m in attr_centroids.items()}
    glob_bands, glob_mat = _as_matrix(global_centroids)

    result = (q_centroids, attr_centroids, global_centroids, by_qkey, question_texts, attr_mats, (glob_bands, glob_mat))
    _save_pickle(cp, result)
    return result

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

def embed_distinct_answers(df: pd.DataFrame, resolved_for_qid: dict[str,str]) -> dict[str, np.ndarray]:
    texts = []
    for _, row in df.iterrows():
        for qid, col in resolved_for_qid.items():
            if col in df.columns:
                a = clean(row.get(col, ""))
                if a: texts.append(a)
    uniq = sorted(set(texts))
    key = hashlib.md5(("\n".join(uniq)).encode("utf-8")).hexdigest()
    cp = _cache_path(f"answer_vecs_{key}")
    cached = _load_pickle(cp)
    if cached is not None: return cached
    if not uniq: return {}
    embedder = get_embedder()
    vecs = embedder.encode(uniq, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    ans2vec = {t:v for t,v in zip(uniq, vecs)}
    _save_pickle(cp, ans2vec)
    return ans2vec

# ==============================
# REQUIRED ONLINE LLM (OpenAI-compatible)
# ==============================
@dataclass
class LLMResult:
    band: int | None
    reason: str
    raw: str

SYSTEM_PROMPT = (
    "You are a careful assessor. Score answers 0–3 using the rubric. "
    "Respond ONLY in compact JSON with keys: band (0..3) and reason (<=50 words)."
)

USER_TMPL = """CASE:
{case}

QUESTION:
{question}

ANSWER:
{answer}

RUBRIC (0–3):
0 = Counterproductive
1 = Compliant
2 = Strategic
3 = Transformative

Return JSON: {{"band":0-3,"reason":"<=50 words"}}.
"""

def _assert_llm_ready():
    if CENTROID_ONLY:
        return
    if not LLM_API_BASE or not LLM_API_KEY or not LLM_MODEL:
        raise RuntimeError("LLM_API_BASE, LLM_API_KEY, and LLM_MODEL must be set in st.secrets (or set CENTROID_ONLY=true).")

def _post_chat(messages, temperature=LLM_TEMPERATURE, max_tokens=160):
    url = f"{LLM_API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "temperature": float(temperature), "max_tokens": int(max_tokens), "messages": messages}
    r = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT_SEC)
    if r.status_code == 429:
        time.sleep(2.0)
        r = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()

def llm_score_via_api(case_text: str, question_text: str, answer_text: str) -> LLMResult:
    if CENTROID_ONLY:
        return LLMResult(None, "LLM disabled (CENTROID_ONLY)", "")
    user_payload = USER_TMPL.format(case=(case_text or "").strip(),
                                    question=(question_text or "").strip(),
                                    answer=(answer_text or "").strip())
    try:
        resp = _post_chat(
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":user_payload}]
        )
        text = resp["choices"][0]["message"]["content"].strip()
        raw = text
        text = text.replace("```json","").replace("```","").strip()
        data = json.loads(text)
        band = int(data.get("band")) if "band" in data else None
        reason = str(data.get("reason","")).strip()
        if band not in (0,1,2,3): return LLMResult(None, "LLM band invalid.", raw)
        return LLMResult(band, reason[:140] if reason else "", raw)
    except Exception as e:
        return LLMResult(None, f"LLM error: {e}", "")

# ==============================
# CONFIDENCE (centroids)
# ==============================
def _softmax_like(scores: dict[int, float]) -> dict[int, float]:
    if not scores: return {}
    vals = list(scores.values()); m = max(vals)
    exps = {k: float(np.exp(v - m)) for k, v in scores.items()}
    s = sum(exps.values())
    return {k: (v/s) for k, v in exps.items()} if s > 0 else {k:0.0 for k in scores}

def _sim_bands(vec: np.ndarray, bands, mat):
    if mat is None: return {}
    sims = mat @ vec  # normalized vectors => cosine
    return {b: float(s) for b, s in zip(bands, sims)}

def _centroid_pick_with_conf(q_sims: dict[int,float], a_sims: dict[int,float], g_sims: dict[int,float]):
    qw = 1.0 if q_sims else 0.0; aw = 0.6 if a_sims else 0.0; gw = 0.4 if g_sims else 0.0
    qp = _softmax_like(q_sims) if q_sims else {}
    ap = _softmax_like(a_sims) if a_sims else {}
    gp = _softmax_like(g_sims) if g_sims else {}
    mix = {b: qw*qp.get(b,0)+aw*ap.get(b,0)+gw*gp.get(b,0) for b in (0,1,2,3)}
    if not mix: return None, 0.0
    band = max(mix, key=mix.get); conf = float(mix[band])
    return band, conf

# ==============================
# SCORER (Preprocess-first, batched LLM)
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, by_qkey, question_texts,
                    attr_mats, glob_pack,
                    case_text: str):

    df_cols = list(df.columns)
    with st.expander("🔎 Advisory-ish columns present", expanded=False):
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

    # --- PREPROCESSING #1: resolve Kobo columns once ---
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]
    resolved_for_qid, missing_map_rows = {}, []
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit: resolved_for_qid[qid] = hit
        else:   missing_map_rows.append((qid, qhint))

    with st.expander("🧭 Mapping → Kobo column resolution", expanded=False):
        if resolved_for_qid:
            show = list(resolved_for_qid.items())[:60]
            st.dataframe(pd.DataFrame(show, columns=["question_id","kobo_column"]))
        if missing_map_rows:
            st.warning(f"{len(missing_map_rows)} question_ids not found (showing up to 30).")
            st.dataframe(pd.DataFrame(missing_map_rows[:30], columns=["question_id","prompt_hint"]))

    # --- PREPROCESSING #2: embed distinct answers (disk cache) ---
    ans2vec = embed_distinct_answers(df, resolved_for_qid)

    # convenience from centroid mats
    attr_mats_local = attr_mats  # {attr: (bands, mat)}
    glob_bands, glob_mat = glob_pack

    # --- PREP LLM WORK QUEUE (first pass: compute centroid + routing decisions) ---
    cells = {}   # key=(row_idx, attr, qn) -> info dict
    rows_out = []
    for i, resp in df.iterrows():
        out = {}
        out["ID"] = (pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
                     if pd.notna(dt_series.iloc[i]) else str(i))
        out["Staff ID"] = str(resp.get(staff_id_col)) if staff_id_col else ""
        out["Duration_min"] = float(duration_min.iloc[i]) if not pd.isna(duration_min.iloc[i]) else ""

        per_attr = {}
        ai_flags = []

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            dfcol = resolved_for_qid.get(qid)
            if not dfcol or dfcol not in df.columns: continue

            ans = clean(resp.get(dfcol, "")); 
            if not ans: continue
            ai_flags.append(looks_ai_like(ans))
            vec = ans2vec.get(ans)
            # only Q1..Q4
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4): continue

            # centroid sims (matrix)
            sims_q = {}
            qkey = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)
            if vec is not None:
                if qkey and qkey in q_centroids:
                    # build once-per-qkey matrix lazily
                    q_bands, q_mat = None, None
                    # cache per-session in st.session_state
                    qcache = st.session_state.setdefault("_qkey_mats", {})
                    if qkey not in qcache:
                        items = [(b,v) for b,v in q_centroids[qkey].items() if v is not None]
                        if items:
                            items.sort(key=lambda x:x[0])
                            q_bands = [b for b,_ in items]
                            q_mat   = np.vstack([v for _,v in items])
                        qcache[qkey] = (q_bands, q_mat)
                    q_bands, q_mat = qcache[qkey]
                    sims_q = _sim_bands(vec, q_bands, q_mat) if q_mat is not None else {}

                bands_a, mat_a = attr_mats_local.get(attr, ([], None))
                sims_a = _sim_bands(vec, bands_a, mat_a) if mat_a is not None else {}
                sims_g = _sim_bands(vec, glob_bands, glob_mat) if glob_mat is not None else {}
            else:
                sims_a = sims_g = {}

            cent_band, cent_conf = _centroid_pick_with_conf(sims_q, sims_a, sims_g)

            qtext_for_llm = (by_qkey.get(qkey, {}) or {}).get("question_text","") if qkey else ""
            if not qtext_for_llm: qtext_for_llm = qhint or ""

            qtext_full = (by_qkey.get(qkey, {}) or {}).get("question_text","")
            overlap = qa_overlap(ans, qtext_full or qhint)
            risky = looks_ai_like(ans)

            # decide if we will call LLM
            use_llm = (not CENTROID_ONLY)
            if cent_band is not None and cent_conf >= SKIP_LLM_IF_CONF and overlap >= SKIP_LLM_MIN_OVLP and not risky:
                use_llm = False

            cells[(i,attr,qn)] = {
                "ans": ans,
                "qtext_for_llm": qtext_for_llm,
                "cent_band": cent_band, "cent_conf": cent_conf,
                "overlap": overlap, "risky": risky,
                "use_llm": use_llm
            }

            # pre-fill columns; final fusion after LLM phase
            sk = f"{attr}_Qn{qn}"
            rk = f"{attr}_Rubric_Qn{qn}"
            ck = f"{attr}_Qn{qn}_Confidence"
            tk = f"{attr}_Qn{qn}_Route"
            rr = f"{attr}_Qn{qn}_RouteReason"
            lk = f"{attr}_LLM_Qn{qn}"
            lr = f"{attr}_LLM_reason_Qn{qn}"
            cb = f"{attr}_Centroid_Qn{qn}"
            cc = f"{attr}_CentroidConfidence_Qn{qn}"

            out.setdefault(sk, ""); out.setdefault(rk, ""); out.setdefault(ck, "")
            out.setdefault(tk, "review"); out.setdefault(rr, "Incomplete signals")
            out.setdefault(lk, ""); out.setdefault(lr, "")
            out.setdefault(cb, cent_band if cent_band is not None else "")
            out.setdefault(cc, round(cent_conf,3) if cent_band is not None else "")

        # defaults for missing cells
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                for suffix in ["", "_Rubric_Qn{n}", "_Qn{n}_Confidence", "_Qn{n}_Route", "_Qn{n}_RouteReason",
                               "_LLM_Qn{n}", "_LLM_reason_Qn{n}", "_Centroid_Qn{n}", "_CentroidConfidence_Qn{n}"]:
                    col = f"{attr}{suffix.replace('{n}', str(qn))}"
                    out.setdefault(col, "")

        out["_per_attr_scores"] = {}   # temp bag
        out["_ai_flags"] = ai_flags
        rows_out.append(out)

    # --- LLM PHASE (batched & parallel) ---
    def _llm_batch_call(jobs):
        results = {}
        if not jobs: return results
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_LLM) as ex:
            futs = {}
            for key, cell in jobs.items():
                futs[ex.submit(llm_score_via_api, case_text, cell["qtext_for_llm"], cell["ans"])] = key
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    results[key] = LLMResult(None, f"LLM error: {e}", "")
        return results

    jobs = {k:v for k,v in cells.items() if v["use_llm"]}
    llm_out = _llm_batch_call(jobs)

    # --- FUSION PHASE (final bands, routes, reasons) ---
    for (i, attr, qn), info in cells.items():
        row = rows_out[i]
        sk = f"{attr}_Qn{qn}"
        rk = f"{attr}_Rubric_Qn{qn}"
        ck = f"{attr}_Qn{qn}_Confidence"
        tk = f"{attr}_Qn{qn}_Route"
        rr = f"{attr}_Qn{qn}_RouteReason"
        lk = f"{attr}_LLM_Qn{qn}"
        lr = f"{attr}_LLM_reason_Qn{qn}"
        cb = f"{attr}_Centroid_Qn{qn}"
        cc = f"{attr}_CentroidConfidence_Qn{qn}"

        cent_band, cent_conf = info["cent_band"], info["cent_conf"]
        overlap, risky = info["overlap"], info["risky"]
        if info["use_llm"]:
            llm = llm_out.get((i,attr,qn), LLMResult(None,"LLM missing",""))
        else:
            llm = LLMResult(cent_band, "High centroid confidence; LLM skipped", "")

        # fusion
        if cent_band is None or llm.band is None:
            final_band = min(cent_band if cent_band is not None else 3,
                             llm.band   if llm.band   is not None else 3) if (cent_band is not None or llm.band is not None) else None
            route, route_reason = "review", "Incomplete signals"
            final_conf = min(cent_conf or 0.0, 0.6)
        else:
            if abs(cent_band - llm.band) > MAX_DISAGREE:
                final_band = min(cent_band, llm.band)
                route, route_reason = "review", "LLM/centroid disagreement"
                final_conf = min(cent_conf, 0.6)
            else:
                final_band = int(round((cent_band + llm.band)/2))
                final_conf = min(cent_conf or 0.0, 0.95)
                route = "auto" if (final_conf >= MIN_CONF_AUTO) else "review"
                route_reason = "High confidence" if route=="auto" else "Low confidence"

        if overlap < MIN_QA_OVERLAP:
            final_band = min(final_band if final_band is not None else 1, 1)
            route, route_reason = "review", "Low Q/A overlap"
            final_conf = min(final_conf, 0.65)

        if risky:
            route, route_reason = "review", "AI-like pattern"
            final_conf = min(final_conf, 0.6)

        if final_band is not None:
            row[sk] = int(final_band)
            row[rk] = BANDS[int(final_band)]
            row[ck] = round(float(final_conf), 3)
            row[tk] = route
            row[rr] = route_reason
            row[lk] = llm.band if llm.band is not None else ""
            row[lr] = llm.reason or ""
            row[cb] = cent_band if cent_band is not None else ""
            row[cc] = round(cent_conf,3) if cent_band is not None else ""
            row["_per_attr_scores"].setdefault(attr, []).append(int(final_band))

    # --- rollups ---
    for row in rows_out:
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = row["_per_attr_scores"].get(attr, [])
            if not scores:
                row[f"{attr}_Avg (0–3)"] = ""
                row[f"{attr}_RANK"] = ""
            else:
                avg = float(np.mean(scores)); band = int(round(avg))
                overall_total += band
                row[f"{attr}_Avg (0–3)"] = round(avg, 2)
                row[f"{attr}_RANK"] = BANDS[band]
        row["Overall Total (0–24)"] = overall_total
        row["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        row["AI_suspected"] = bool(any(row["_ai_flags"]))
        # cleanup
        del row["_per_attr_scores"]; del row["_ai_flags"]

    res_df = pd.DataFrame(rows_out)

    def order_cols(cols):
        ordered = ["ID","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [
                    f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}",
                    f"{attr}_Qn{qn}_Confidence", f"{attr}_Qn{qn}_Route", f"{attr}_Qn{qn}_RouteReason",
                    f"{attr}_LLM_Qn{qn}", f"{attr}_LLM_reason_Qn{qn}",
                    f"{attr}_Centroid_Qn{qn}", f"{attr}_CentroidConfidence_Qn{qn}",
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
# EXPORTS / SHEETS / STAR SCHEMA
# ==============================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return bio.getvalue()

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
    try:
        w.freeze(rows=1)
        w.spreadsheet.batch_update({"requests":[{"autoResizeDimensions":{
            "dimensions":{"sheetId":w.id,"dimension":"COLUMNS","startIndex":0,"endIndex":len(header)}
        }}]})
    except Exception:
        pass

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
    submission_out = submission[["ID","date_key","staff_key","Duration_min","Overall Total (0–24)","Overall Rank","AI_suspected"]]
    return {"fact_attribute":fact_attribute,"fact_question":fact_question,"dim_staff":dim_staff,"dim_attribute":dim_attribute,"dim_date":dim_date,"submission":submission_out}

# ==============================
# UI / MAIN
# ==============================
def main():
    st.title("📊 Advisory Scoring — Hybrid (Centroids + Online LLM) → Router → Sheets")
    st.caption(f"MIN_CONF_AUTO={MIN_CONF_AUTO} · MAX_DISAGREE={MAX_DISAGREE} · Model: {LLM_MODEL or 'unset'} · CENTROID_ONLY={CENTROID_ONLY}")

    # Ensure LLM is configured or explicitly disabled
    try:
        _assert_llm_ready()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Case text
    case_text = (CASE1_TEXT or "").strip()
    if not case_text:
        st.error("CASE1 is missing or empty in st.secrets. Add the case study text under the key CASE1.")
        st.stop()

    # Optional sample parameter for dev speed (?sample=30)
    try:
        qp = st.query_params
        sample_n = int(qp.get("sample", [0])[0]) if qp and qp.get("sample") else None
    except Exception:
        sample_n = None

    AUTO_RUN  = bool(st.secrets.get("AUTO_RUN", False))
    PUSH_STAR = bool(st.secrets.get("PUSH_STAR_SCHEMA", True))

    def run_pipeline():
        with st.spinner("1) Loading mapping…"):
            mapping = load_mapping_from_path(MAPPING_PATH)

        with st.spinner("2) Fetching Kobo submissions…"):
            df = fetch_kobo_dataframe(sample_n=sample_n)
        if df.empty:
            st.warning("No Kobo submissions found.")
            st.stop()
        st.caption("Fetched sample:"); st.dataframe(df.head(), use_container_width=True)

        # Preprocessing-first: resolve columns + embed unique answers BEFORE centroids
        # (answer cache doesn't depend on exemplars; speeds up iteration)
        df_cols = list(df.columns)
        resolved_for_qid = {}
        all_map = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]
        for r in all_map:
            qid   = r["question_id"]; qhint = r.get("prompt_hint","")
            hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
            if hit: resolved_for_qid[qid] = hit
        with st.spinner("3) Pre-embedding distinct answers…"):
            _ = embed_distinct_answers(df, resolved_for_qid)

        with st.spinner("4) Building semantic centroids… (cached)"):
            q_c, a_c, g_c, by_q, qtexts, attr_mats, glob_pack = _build_centroids_cached(read_jsonl_path(EXEMPLARS_PATH))

        with st.spinner("5) Scoring with hybrid + routing…"):
            scored_df = score_dataframe(df, mapping, q_c, by_q, qtexts, attr_mats, glob_pack, case_text)

        autos_df, review_df = split_auto_review(scored_df)
        st.success("✅ Done.")
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

    if AUTO_RUN and not st.session_state.get("auto_ran_once"):
        st.session_state["auto_ran_once"] = True
        run_pipeline()

    if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True):
        run_pipeline()

    if "scored_df" in st.session_state and st.session_state["scored_df"] is not None:
        with st.expander("📤 Google Sheets export", expanded=True):
            st.write("Spreadsheet key:", st.secrets.get("GSHEETS_SPREADSHEET_KEY") or "⚠️ Not set")
            st.write("Worksheet name:", DEFAULT_WS_NAME)

            def do_push():
                ws = _open_ws_by_key(); sh = ws.spreadsheet
                push_sheet_tab(sh, DEFAULT_WS_NAME, st.session_state["scored_df"])
                push_sheet_tab(sh, "autos_accepted", st.session_state["autos_df"])
                push_sheet_tab(sh, "review_queue", st.session_state["review_df"])
                if PUSH_STAR:
                    st.info("Building star schema…")
                    for title, tdf in build_star_schema_from_scored(st.session_state["scored_df"]).items():
                        push_sheet_tab(sh, title, tdf)
                st.success("✅ Pushed scored, autos_accepted, review_queue" + (" + star schema" if PUSH_STAR else "") + " to Google Sheets.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Push all to Google Sheets", use_container_width=True):
                    try: do_push()
                    except Exception as e: st.error(f"Push failed: {e}")
            with col2:
                st.caption(f"CENTROID_ONLY={CENTROID_ONLY} · SKIP_LLM_IF_CONF={SKIP_LLM_IF_CONF} · MAX_PARALLEL_LLM={MAX_PARALLEL_LLM}")

if __name__ == "__main__":
    main()
