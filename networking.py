import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

import json, re, unicodedata
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# ==============================
# CONSTANTS / PATHS
# ==============================
KOBO_BASE         = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID3    = st.secrets.get("KOBO_ASSET_ID3", "")
KOBO_TOKEN        = st.secrets.get("KOBO_TOKEN", "")
AUTO_PUSH         = bool(st.secrets.get("AUTO_PUSH", False))
AUTO_RUN          = bool(st.secrets.get("AUTO_RUN", True))

DATASETS_DIR      = Path("DATASETS")
MAPPING_PATH      = DATASETS_DIR / "mapping3.csv"
EXEMPLARS_PATH    = DATASETS_DIR / "networking.jsonl"

BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Localization Champion", 21, 24),
    ("Skilled Networked Advocate", 16, 20),
    ("Developing Influencer", 10, 15),
    ("Needs Support", 0, 9),
]

ORDERED_ATTRS = [
    "Strategic Positioning & Donor Fluency",
    "Power-Aware Stakeholder Mapping",
    "Equitable Allyship & Local Fronting",
    "Coalition Governance & Convening",
    "Community-Centered Messaging",
    "Evidence-Led Learning (Local Knowledge)",
    "Influence Without Authority",
    "Risk Management & Adaptive Communication",
]

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

PASSTHROUGH_HINTS = [
    "staff id","staff_id","staffid","_id","id","_uuid","uuid","instanceid","_submission_time",
    "submissiondate","submission_date","start","_start","end","_end","today","date","deviceid",
    "username","enumerator","submitted_via_web","_xform_id_string","formid","assetid","care_staff"
]
_EXCLUDE_SOURCE_COLS_LOWER = {
    "_id","formhub/uuid","start","end","today","staff_id","meta/instanceid",
    "_xform_id_string","_uuid","meta/rootuuid","_submission_time","_validation_status"
}

# ==============================
# AI DETECTION
# ==============================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

TRANSITION_OPEN_RX = re.compile(r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b", re.I)
LIST_CUES_RX       = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX          = re.compile(r"^[-*•]\s", re.M)
SYMBOL_RX          = re.compile(r"[—–\-]{2,}|[≥≤≧≦≈±×÷%]|[→←⇒↔↑↓]|[•●◆▶✓✔✗❌§†‡]", re.U)
TIMEBOX_RX         = re.compile(r"(?:\bday\s*\d+\b|\bweek\s*\d+\b|\bmonth\s*\d+\b|\bquarter\s*\d+\b|\b\d+\s*(?:days?|weeks?|months?|quarters?)\b|\bby\s+day\s*\d+\b)", re.I)
EXPLICIT_AI_RX     = re.compile(r"(?:as an ai\b|i am an ai\b)", re.I)

def clean(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+"," ", s).strip()

def qa_overlap(ans: str, qtext: str) -> float:
    at = set(re.findall(r"\w+", (ans or "").lower()))
    qt = set(re.findall(r"\w+", (qtext or "").lower()))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

def _avg_sentence_len(text: str) -> float:
    sents = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    if not sents: return 0.0
    toks = re.findall(r"\w+", text or "")
    return len(toks) / max(len(sents), 1)

def _type_token_ratio(text: str) -> float:
    toks = [t.lower() for t in re.findall(r"[a-z]+", text or "")]
    return 1.0 if not toks else len(set(toks))/len(toks)

def ai_signal_score(text: str, question_hint: str = "") -> float:
    t = clean(text)
    if not t: return 0.0
    score = 0.0
    if SYMBOL_RX.search(t):                 score += 0.35
    if TIMEBOX_RX.search(t):                score += 0.15
    if EXPLICIT_AI_RX.search(t):            score += 0.35
    if TRANSITION_OPEN_RX.search(t):        score += 0.12
    if LIST_CUES_RX.search(t):              score += 0.12
    if BULLET_RX.search(t):                 score += 0.08
    buzz_hits = sum(1 for b in {
        "minimum viable","feedback loop","trade-off","evidence-based","stakeholder alignment",
        "learners’ agency","norm shifts","quick win","low-lift","scalable","best practice",
        "pilot theatre","timeboxed"
    } if b in t.lower())
    if buzz_hits:                            score += min(0.24, 0.08*buzz_hits)
    asl = _avg_sentence_len(t)
    if   asl >= 26:                          score += 0.18
    elif asl >= 18:                          score += 0.10
    ttr = _type_token_ratio(t)
    if ttr <= 0.45 and len(t) >= 180:        score += 0.10
    if question_hint and qa_overlap(t, question_hint) < 0.06:
                                             score += 0.10
    return max(0.0, min(1.0, score))

# ==============================
# KOBO
# ==============================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID3 or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID3 and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions","data"):
        url = kobo_url(KOBO_ASSET_ID3, kind)
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
    st.error("Could not fetch data. Check KOBO_BASE, KOBO_ASSET_ID3, token permissions.")
    return pd.DataFrame()

# ==============================
# MAPPING + EXEMPLARS
# ==============================
QID_PREFIX_TO_SECTION = {"SPD":"A1","PAS":"A2","EAL":"A3","CGC":"A4","CCM":"A5","ELL":"A6","IWA":"A7","RMA":"A8"}
QNUM_RX = re.compile(r"_Q(\d+)$")

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

def build_kobo_bases_from_qid(question_id: str) -> list[str]:
    if not question_id: return []
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m: return []
    qn = m.group(1)
    prefix = qid.split("_Q")[0]
    if prefix not in QID_PREFIX_TO_SECTION: return []
    section = QID_PREFIX_TO_SECTION[prefix]
    roots = ["networking", "Networking"]
    return [f"{root}/{section}_Section/{section}_{qn}" for root in roots]

def expand_possible_kobo_columns(base: str) -> list[str]:
    if not base: return []
    return [
        base,
        f"{base} :: Answer (text)",
        f"{base} :: English (en)",
        f"{base} - English (en)",
        f"{base}_labels",
        f"{base}_label",
    ]

def _score_kobo_header(col: str, token: str) -> int:
    c = col.lower(); t = token.lower()
    if c == t: return 100
    s = 0
    if c.endswith("/" + t): s = max(s, 95)
    if f"/{t}/" in c: s = max(s, 92)
    if (f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c): s = max(s, 90)
    if t in c: s = max(s, 80)
    if "english" in c or "label" in c or "(en)" in c: s += 5
    if "answer (text)" in c or "answer_text" in c or "text" in c: s += 5
    if "growthmindset/" in c or "networking/" in c or "/a" in c: s += 2
    return s

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    cols_original = list(df_cols)
    cols_lower_map = {c.lower(): c for c in cols_original}

    bases = build_kobo_bases_from_qid(question_id)
    for base in bases:
        if base.lower() in cols_lower_map:
            return cols_lower_map[base.lower()]
        for v in expand_possible_kobo_columns(base):
            if v.lower() in cols_lower_map:
                return cols_lower_map[v.lower()]
        for c in cols_original:
            if c.lower().startswith(base.lower()):
                return c

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

    if token:
        best_col, best_score = None, 0
        for col in cols_original:
            s = _score_kobo_header(col, token)
            if s > best_score:
                best_score, best_col = s, col
        if best_col and best_score >= 82:
            return best_col

    hint = clean(prompt_hint or "")
    if hint:
        candidates = [(c, c.lower()) for c in cols_original]
        hits = process.extract(hint.lower(), [lo for _, lo in candidates],
                               scorer=fuzz.partial_token_set_ratio, limit=5)
        for _, lo, score in hits:
            if score >= 88:
                for orig, low in candidates:
                    if low == lo:
                        return orig
    return None

# ==============================
# EMBEDDINGS / CENTROIDS
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_centroids_cached(exemplars_tuple: tuple) -> tuple:
    exemplars = list(exemplars_tuple)
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

# Fast embedding store
_EMB_STORE: dict[str, np.ndarray] = {}

@st.cache_resource(show_spinner=False)
def _embed_texts_cached(texts_tuple: tuple[str, ...]) -> dict[str, np.ndarray]:
    texts = [t for t in texts_tuple if t]
    if not texts:
        return {}
    embs = get_embedder().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return {t: e for t, e in zip(texts, embs)}

def embed_many(texts: list[str]) -> None:
    missing = [t for t in texts if t and t not in _EMB_STORE]
    if not missing:
        return
    _EMB_STORE.update(_embed_texts_cached(tuple(missing)))

def emb_of(text: str):
    t = clean(text)
    if not t:
        return None
    v = _EMB_STORE.get(t)
    if v is not None:
        return v
    pack = _embed_texts_cached((t,))
    _EMB_STORE.update(pack)
    return _EMB_STORE.get(t)

# ==============================
# SCORING (optimized)
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_centroids, attr_centroids, global_centroids,
                    by_qkey, question_texts):

    df_cols = list(df.columns)

    with st.expander("🔎 Debug: `Networking` section columns present", expanded=False):
        sample_cols = [c for c in df_cols if "/A" in c or "Networking/" in c or c.startswith("A")]
        st.write(sample_cols[:80])

    staff_id_col = next((c for c in df.columns if c.strip().lower() in ("staff id","staff_id")), None)
    staff_id_idx = (df.columns.get_loc(staff_id_col) if staff_id_col in df.columns else None)

    date_cols_pref = ["_submission_time","SubmissionDate","submissiondate","end","End","start","Start","today","date","Date"]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])
    # Precompute times once
    dt_series = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series([pd.NaT]*len(df))

    start_col = next((c for c in ["start","Start","_start"] if c in df.columns), None)
    end_col   = next((c for c in ["end","End","_end","_submission_time","SubmissionDate","submissiondate"] if c in df.columns), None)
    start_dt  = pd.to_datetime(df[start_col], errors="coerce") if start_col else pd.Series([pd.NaT]*len(df))
    end_dt    = pd.to_datetime(df[end_col], errors="coerce")   if end_col   else pd.Series([pd.NaT]*len(df))
    duration_min = ((end_dt - start_dt).dt.total_seconds()/60.0).round(2)

    # Prepare mapping rows
    mapping_rows = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]

    # Resolve Kobo column once per qid and store direct column index for fast row access
    resolved_for_qid: dict[str, str] = {}
    col_index_for_qid: dict[str, int] = {}
    missing_map_rows = []
    for r in mapping_rows:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit:
            resolved_for_qid[qid] = hit
            try:
                col_index_for_qid[qid] = df.columns.get_loc(hit)
            except KeyError:
                pass
        else:
            missing_map_rows.append((qid, qhint))

    with st.expander("🧭 Mapping → Kobo column resolution (by question_id)", expanded=False):
        if resolved_for_qid:
            show = list(resolved_for_qid.items())[:60]
            st.dataframe(pd.DataFrame(show, columns=["question_id","kobo_column"]))
        if missing_map_rows:
            st.warning(f"{len(missing_map_rows)} question_ids not found in Kobo headers (showing up to 30).")
            st.dataframe(pd.DataFrame(missing_map_rows[:30], columns=["question_id","prompt_hint"]))

    # Resolve qkey ONCE per question
    qkey_for_qid = {}
    qtext_for_qkey = {k: v.get("question_text","") for k, v in by_qkey.items()}
    for r in mapping_rows:
        qid, qhint = r["question_id"], r.get("prompt_hint","")
        qkey_for_qid[qid] = resolve_qkey(q_centroids, by_qkey, question_texts, qid, qhint)

    # Collect ALL distinct answers once (with length cap for speed)
    LIMIT_EMB_CHARS = 1000  # safe cap; models truncate long tails anyway
    distinct_answers = set()
    for r in mapping_rows:
        qid = r["question_id"]
        if qid not in col_index_for_qid:
            continue
        cidx = col_index_for_qid[qid]
        col_vals = (df.iloc[:, cidx].astype(str).str.slice(0, LIMIT_EMB_CHARS).map(clean)).tolist()
        distinct_answers.update([t for t in col_vals if t])

    # Batch embed once
    embed_many(list(distinct_answers))

    def best_sim(vec, cent_dict):
        if not cent_dict or vec is None:
            return None
        best_s, best_v = None, -1e9
        for s, v in cent_dict.items():
            if v is None: continue
            sim = float(np.dot(vec, v))
            if sim > best_v:
                best_v, best_s = sim, s
        return best_s

    rows_out = []
    # Fast row traversal: itertuples(name=None) yields a tuple in column order
    for i, row_t in enumerate(df.itertuples(index=False, name=None)):
        out = {}
        # fast pulls
        dt_val = dt_series.iat[i] if len(dt_series) else pd.NaT
        out["Date"] = (pd.to_datetime(dt_val).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_val) else str(i))
        out["Staff ID"] = (str(row_t[staff_id_idx]) if staff_id_idx is not None else "")
        dur = duration_min.iat[i] if len(duration_min) else np.nan
        out["Duration_min"] = float(dur) if not pd.isna(dur) else ""

        per_attr = {}
        ai_scores = []

        for r in mapping_rows:
            qid, attr = r["question_id"], r["attribute"]
            qhint     = r.get("prompt_hint","")
            cidx      = col_index_for_qid.get(qid, None)
            if cidx is None:
                continue

            ans = clean(str(row_t[cidx]))[:LIMIT_EMB_CHARS]

            # figure out qn
            qn = None
            if "_Q" in (qid or ""):
                try: qn = int(qid.split("_Q")[-1])
                except: qn = None
            if qn not in (1,2,3,4):
                continue

            score_key  = f"{attr}_Qn{qn}"
            rubric_key = f"{attr}_Rubric_Qn{qn}"

            if not ans:
                # guarantee filled cell
                out[score_key]  = 0
                out[rubric_key] = BANDS[0]
                per_attr.setdefault(attr, []).append(0)
                continue

            vec  = emb_of(ans)
            qkey = qkey_for_qid.get(qid)

            sc = None
            if qkey and qkey in q_centroids:
                sc = best_sim(vec, q_centroids[qkey])
            if sc is None and attr in attr_centroids:
                sc = best_sim(vec, attr_centroids[attr])
            if sc is None:
                sc = best_sim(vec, global_centroids)

            # mismatch penalty by overlap
            qtext = qtext_for_qkey.get(qkey, "") if qkey else qhint
            if sc is not None and qa_overlap(ans, qtext or qhint) < MIN_QA_OVERLAP:
                sc = min(sc, 1)

            if sc is None:
                sc = 0

            out[score_key]  = int(sc)
            out[rubric_key] = BANDS[int(sc)]
            per_attr.setdefault(attr, []).append(int(sc))

            # AI suspicion for this answer
            ai_scores.append(ai_signal_score(ans, qtext))

        # ensure fixed shape for per-question blocks
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                out.setdefault(f"{attr}_Qn{qn}", "")
                out.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + ranks
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
        out["AI-Suspected"] = bool(any(s >= AI_SUSPECT_THRESHOLD for s in ai_scores))
        rows_out.append(out)

    res_df = pd.DataFrame(rows_out)

    def order_cols(cols):
        ordered = ["Date","Staff ID","Duration_min"]
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                ordered += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
        for attr in ORDERED_ATTRS:
            ordered += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
        ordered += ["Overall Total (0–24)", "Overall Rank", "AI-Suspected"]
        extras = [c for c in cols if c not in ordered]
        return [c for c in ordered if c in cols] + extras

    return res_df.reindex(columns=order_cols(list(res_df.columns)))

# ==============================
# EXPORTS / SHEETS
# ==============================
def _ensure_ai_last(df: pd.DataFrame,
                    export_name: str = "AI-Suspected",
                    legacy_name: str = "AI_suspected") -> pd.DataFrame:
    out = df.copy()
    if export_name not in out.columns and legacy_name in out.columns:
        out = out.rename(columns={legacy_name: export_name})
    if export_name not in out.columns:
        out[export_name] = ""
    cols = [c for c in out.columns if c != export_name] + [export_name]
    return out[cols]

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    df_out = _ensure_ai_last(df)
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False)
    return bio.getvalue()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME3", "Networking")

def _normalize_sa_dict(raw: dict) -> dict:
    if not raw:
        raise ValueError("gcp_service_account missing in secrets.")
    sa = dict(raw)
    if "token_ur" in sa and "token_uri" not in sa:
        sa["token_uri"] = sa.pop("token_ur")
    if sa.get("private_key") and "\\n" in sa["private_key"]:
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    sa.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    sa.setdefault("auth_uri", "https://accounts.google.com/o/oauth2/auth")
    sa.setdefault("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs")
    required = ["type","project_id","private_key_id","private_key","client_email","client_id","token_uri"]
    missing = [k for k in required if not sa.get(k)]
    if missing:
        raise ValueError(f"gcp_service_account missing fields: {', '.join(missing)}")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets.get("gcp_service_account"))
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

def _open_ws_by_key() -> gspread.Worksheet:
    key = st.secrets.get("GSHEETS_SPREADSHEET_KEY")
    ws_name = DEFAULT_WS_NAME
    if not key:
        raise ValueError("GSHEETS_SPREADSHEET_KEY not set in secrets.")
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

def _to_a1_col(n: int) -> str:
    s = []
    while n > 0:
        n, r = divmod(n - 1, 26)
        s.append(chr(65 + r))
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

def upload_df_to_gsheets(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        df = _ensure_ai_last(df)
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
# PAGE ENTRYPOINT
# ==============================
def main():
    st.title("📊 Net Working and Advocacy: Kobo → Scored Excel / Google Sheets")

    def run_pipeline():
        try:
            mapping = load_mapping_from_path(MAPPING_PATH)
        except Exception as e:
            st.error(f"Failed to load mapping from {MAPPING_PATH}: {e}")
            return

        try:
            exemplars = read_jsonl_path(EXEMPLARS_PATH)
            if not exemplars:
                st.error(f"Exemplars file is empty: {EXEMPLARS_PATH}")
                return
        except Exception as e:
            st.error(f"Failed to read exemplars from {EXEMPLARS_PATH}: {e}")
            return

        with st.spinner("Building semantic centroids..."):
            q_centroids, attr_centroids, global_centroids, by_qkey, question_texts = build_centroids_cached(tuple(exemplars))

        with st.spinner("Fetching Kobo submissions..."):
            df = fetch_kobo_dataframe()
        if df.empty:
            st.warning("No Kobo submissions found.")
            return

        st.caption("Fetched sample:")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("Scoring responses..."):
            scored_df = score_dataframe(df, mapping, q_centroids, attr_centroids, global_centroids, by_qkey, question_texts)

        st.success("✅ Scoring complete.")
        st.dataframe(scored_df.head(50), use_container_width=True)
        st.session_state["scored_df"] = scored_df

        st.download_button(
            "⬇️ Download Excel",
            data=to_excel_bytes(scored_df),
            file_name="Networking_Scoring.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        if AUTO_PUSH:
            with st.spinner("📤 Sending to Google Sheets..."):
                ok, msg = upload_df_to_gsheets(scored_df)
            (st.success if ok else st.error)(msg)

    if AUTO_RUN:
        run_pipeline()
    else:
        if st.button("🚀 Fetch Kobo & Score", type="primary", use_container_width=True):
            run_pipeline()

    if ("scored_df" in st.session_state and
        st.session_state["scored_df"] is not None and
        not AUTO_PUSH):
        with st.expander("Google Sheets export", expanded=True):
            st.write("Spreadsheet key:", st.secrets.get("GSHEETS_SPREADSHEET_KEY") or "⚠️ Not set")
            st.write("Worksheet name:", DEFAULT_WS_NAME)
            if st.button("📤 Send scored table to Google Sheets", use_container_width=True):
                ok, msg = upload_df_to_gsheets(st.session_state["scored_df"])
                if ok: st.success(msg)
                else:  st.error(msg)

if __name__ == "__main__":
    main()
