# dashboard.py
# ------------------------------------------------------------
# PowerBI-like dashboard for scored rubrics (Google Sheets)
# Pages: Thought Leadership, Growth Mindset, Networking, Advisory, Influencingrelationship
# Tabs inside each page: each Attribute/Section + Overall
# Charts: column, bar, donut/pie, histogram, heatmaps
# Robust to missing columns (skips gracefully)
# ------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Scoring Dashboard", layout="wide")

# -----------------------------
# Secrets / Google Sheets
# -----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

GSHEETS_SPREADSHEET_KEY = st.secrets.get("GSHEETS_SPREADSHEET_KEY", "")
# Your worksheet names (as you described)
GSHEETS_WORKSHEET_NAME  = st.secrets.get("GSHEETS_WORKSHEET_NAME",  "Advisory")
GSHEETS_WORKSHEET_NAME1 = st.secrets.get("GSHEETS_WORKSHEET_NAME1", "Thought Leadership")
GSHEETS_WORKSHEET_NAME2 = st.secrets.get("GSHEETS_WORKSHEET_NAME2", "Growth Mindset")
GSHEETS_WORKSHEET_NAME3 = st.secrets.get("GSHEETS_WORKSHEET_NAME3", "Networking")
GSHEETS_WORKSHEET_NAME4 = st.secrets.get("GSHEETS_WORKSHEET_NAME4", "Influencingrelationship")

WORKSHEETS = {
    "Thought Leadership": GSHEETS_WORKSHEET_NAME1,
    "Growth Mindset": GSHEETS_WORKSHEET_NAME2,
    "Networking & Advocacy": GSHEETS_WORKSHEET_NAME3,
    "Advisory": GSHEETS_WORKSHEET_NAME,
    "Influencing Relationship": GSHEETS_WORKSHEET_NAME4,
}

# -----------------------------
# Helpers
# -----------------------------
def _normalize_sa_dict(raw: dict) -> dict:
    if not raw:
        raise ValueError("gcp_service_account missing in st.secrets.")
    sa = dict(raw)
    # common typo
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

@st.cache_data(ttl=120, show_spinner=False)
def load_sheet_as_df(sheet_name: str) -> pd.DataFrame:
    if not GSHEETS_SPREADSHEET_KEY:
        return pd.DataFrame()

    gc = gs_client()
    sh = gc.open_by_key(GSHEETS_SPREADSHEET_KEY)
    ws = sh.worksheet(sheet_name)

    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()

    header = [str(x).strip() for x in values[0]]
    rows = values[1:]

    df = pd.DataFrame(rows, columns=header)

    # strip whitespace
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # numeric conversions: Qn cols, Avg, Overall Total, AI_MaxScore
    num_cols = [c for c in df.columns if re.search(r"_Qn\d+$", c)]
    num_cols += [c for c in df.columns if "Avg" in c or "Overall Total" in c or c in ("AI_MaxScore",)]
    for c in set(num_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace({"": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")

    # boolean conversion for AI flags if present
    for ai_col in ["AI_Suspected", "AI-Suspected"]:
        if ai_col in df.columns:
            df[ai_col] = df[ai_col].astype(str).str.upper().isin(["TRUE","1","YES","Y"])

    return df

def clean_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    drop_like = {"date", "duration", "duration_min", "start", "end"}
    keep_cols = []
    for c in df.columns:
        cl = c.strip().lower()
        if any(k in cl for k in drop_like):
            continue
        keep_cols.append(c)
    return df[keep_cols].copy()

def parse_attributes(df: pd.DataFrame):
    """
    Detect attributes dynamically from columns:
      <Attr>_Qn1 ... <Attr>_Rubric_Qn1
      <Attr>_Avg (0–3), <Attr>_RANK
    Returns dict: attr -> {"qns":[1..], "score_cols":{qn:col}, "rubric_cols":{qn:col}, "avg":col/None, "rank":col/None}
    """
    out = {}
    if df.empty:
        return out

    qn_rx = re.compile(r"^(.*)_Qn(\d+)$")
    rb_rx = re.compile(r"^(.*)_Rubric_Qn(\d+)$")

    for c in df.columns:
        m = qn_rx.match(c)
        if m:
            attr, qn = m.group(1).strip(), int(m.group(2))
            out.setdefault(attr, {"qns": set(), "score_cols": {}, "rubric_cols": {}, "avg": None, "rank": None})
            out[attr]["qns"].add(qn)
            out[attr]["score_cols"][qn] = c

        m2 = rb_rx.match(c)
        if m2:
            attr, qn = m2.group(1).strip(), int(m2.group(2))
            out.setdefault(attr, {"qns": set(), "score_cols": {}, "rubric_cols": {}, "avg": None, "rank": None})
            out[attr]["qns"].add(qn)
            out[attr]["rubric_cols"][qn] = c

    # avg/rank
    for attr in list(out.keys()):
        avg_candidates = [c for c in df.columns if c.startswith(attr) and "Avg" in c]
        rank_candidates = [c for c in df.columns if c.startswith(attr) and c.endswith("_RANK")]
        out[attr]["avg"] = avg_candidates[0] if avg_candidates else None
        out[attr]["rank"] = rank_candidates[0] if rank_candidates else None
        out[attr]["qns"] = sorted(list(out[attr]["qns"]))

    # stable ordering: keep as they appear in df (approx powerbi feel)
    ordered = []
    for c in df.columns:
        m = qn_rx.match(c)
        if m:
            a = m.group(1).strip()
            if a in out and a not in ordered:
                ordered.append(a)
    # add any remaining
    for a in out.keys():
        if a not in ordered:
            ordered.append(a)

    return {a: out[a] for a in ordered}

def score_distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Percent distribution for 0-3 scores."""
    if col not in df.columns:
        return pd.DataFrame(columns=["Score","Count","Percent"])
    s = df[col].dropna()
    counts = s.value_counts().reindex([0,1,2,3], fill_value=0).reset_index()
    counts.columns = ["Score","Count"]
    total = counts["Count"].sum() or 1
    counts["Percent"] = (counts["Count"] / total * 100).round(1)
    return counts

def rubric_distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["Rubric","Count"])
    s = df[col].replace({"": np.nan, "None": np.nan}).dropna().astype(str)
    counts = s.value_counts().reset_index()
    counts.columns = ["Rubric","Count"]
    return counts

def heatmap_qn_by_score(df: pd.DataFrame, score_cols: dict) -> pd.DataFrame:
    """
    Build a matrix: rows=Qn, cols=Score(0-3), values=count
    """
    rows = []
    for qn, col in sorted(score_cols.items()):
        if col not in df.columns:
            continue
        dist = score_distribution(df, col).set_index("Score")["Count"].to_dict()
        rows.append({"Qn": f"Qn{qn}", **{str(k): int(dist.get(k, 0)) for k in [0,1,2,3]}})
    if not rows:
        return pd.DataFrame()
    h = pd.DataFrame(rows).set_index("Qn")
    return h[["0","1","2","3"]]

def overall_block(df: pd.DataFrame):
    # handle both naming styles
    ai_col = "AI_Suspected" if "AI_Suspected" in df.columns else ("AI-Suspected" if "AI-Suspected" in df.columns else None)
    overall_total = next((c for c in df.columns if c.startswith("Overall Total")), None)
    overall_rank  = "Overall Rank" if "Overall Rank" in df.columns else None
    ai_maxscore   = "AI_MaxScore" if "AI_MaxScore" in df.columns else None

    return ai_col, overall_total, overall_rank, ai_maxscore

# -----------------------------
# UI Header
# -----------------------------
st.title("Scored Rubric Dashboard")
st.caption("PowerBI-like tabs per section, pulling scored results directly from Google Sheets.")

if not GSHEETS_SPREADSHEET_KEY:
    st.error("GSHEETS_SPREADSHEET_KEY is missing in st.secrets.")
    st.stop()

# -----------------------------
# Sidebar: choose page (worksheet)
# -----------------------------
page = st.sidebar.radio("Dashboard Page", list(WORKSHEETS.keys()), index=0)
sheet_name = WORKSHEETS[page]

with st.sidebar.expander("Google Sheets Source", expanded=False):
    st.write("Spreadsheet key:", GSHEETS_SPREADSHEET_KEY)
    st.write("Worksheet:", sheet_name)

# -----------------------------
# Load data
# -----------------------------
with st.spinner(f"Loading '{sheet_name}' from Google Sheets..."):
    raw = load_sheet_as_df(sheet_name)
df = clean_dashboard_df(raw)

if df.empty:
    st.warning(f"No rows found in worksheet '{sheet_name}'.")
    st.stop()

attrs = parse_attributes(df)
ai_col, overall_total_col, overall_rank_col, ai_maxscore_col = overall_block(df)

# -----------------------------
# Create tabs: each attribute + Overall
# -----------------------------
tab_names = list(attrs.keys()) + ["Overall"]
tabs = st.tabs(tab_names)

# -----------------------------
# Attribute tabs
# -----------------------------
for idx, attr in enumerate(attrs.keys()):
    meta = attrs[attr]
    with tabs[idx]:
        st.subheader(attr)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Responses", f"{len(df):,}")

        if meta["avg"] and meta["avg"] in df.columns:
            c2.metric("Avg (0–3)", f"{pd.to_numeric(df[meta['avg']], errors='coerce').mean():.2f}")
        else:
            c2.metric("Avg (0–3)", "—")

        if meta["rank"] and meta["rank"] in df.columns:
            top_rank = df[meta["rank"]].replace({"": np.nan}).dropna().astype(str).value_counts().head(1)
            c3.metric("Most common rank", top_rank.index[0] if len(top_rank) else "—")
        else:
            c3.metric("Most common rank", "—")

        # % Transformative on this attribute (using Avg if present, else mean of Qns)
        def _attr_transformative_rate():
            # treat "Transformative" = 3, "Strategic"=2 etc; use per-question scores
            cols = [meta["score_cols"][qn] for qn in meta["qns"] if qn in meta["score_cols"] and meta["score_cols"][qn] in df.columns]
            if not cols:
                return None
            vals = df[cols].apply(pd.to_numeric, errors="coerce")
            # percentage of all filled cells that are 3
            filled = vals.notna().sum().sum()
            if filled == 0:
                return 0.0
            return float((vals == 3).sum().sum() / filled * 100)

        tr = _attr_transformative_rate()
        c4.metric("% Transformative (Qns)", f"{tr:.1f}%" if tr is not None else "—")

        st.divider()

        # Heatmap: Qn vs Score distribution counts
        h = heatmap_qn_by_score(df, meta["score_cols"])
        if not h.empty:
            st.markdown("**Heatmap: counts by Question vs Score (0–3)**")
            fig_h = px.imshow(
                h,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Score", y="Question", color="Count"),
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No question score columns found for this section (skipping heatmap).")

        st.divider()

        # Per-question charts
        for qn in meta["qns"]:
            score_col = meta["score_cols"].get(qn)
            rub_col = meta["rubric_cols"].get(qn)

            if (score_col not in df.columns) and (rub_col not in df.columns):
                continue

            st.markdown(f"### Qn{qn}")

            left, right = st.columns(2)

            # Column chart: % by numeric score
            if score_col in df.columns:
                dist = score_distribution(df, score_col)
                fig = px.bar(dist, x="Score", y="Percent", text="Percent")
                fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
                fig.update_layout(yaxis_title="Percent", xaxis_title="Score (0–3)")
                left.plotly_chart(fig, use_container_width=True)
            else:
                left.info("Missing numeric score column for this question (skipping % chart).")

            # Bar chart: rubric frequency
            if rub_col in df.columns:
                rdist = rubric_distribution(df, rub_col)
                fig2 = px.bar(rdist, x="Count", y="Rubric", orientation="h", text="Count")
                fig2.update_layout(xaxis_title="Count", yaxis_title="")
                right.plotly_chart(fig2, use_container_width=True)
            else:
                right.info("Missing rubric column for this question (skipping rubric chart).")

        # If nothing rendered
        if not meta["qns"]:
            st.info("No questions detected for this attribute.")

# -----------------------------
# Overall tab
# -----------------------------
with tabs[-1]:
    st.subheader("Overall Analysis")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responses", f"{len(df):,}")

    if overall_total_col and overall_total_col in df.columns:
        c2.metric("Mean Overall Total", f"{pd.to_numeric(df[overall_total_col], errors='coerce').mean():.2f}")
    else:
        c2.metric("Mean Overall Total", "—")

    if overall_rank_col and overall_rank_col in df.columns:
        top = df[overall_rank_col].replace({"": np.nan}).dropna().astype(str).value_counts().head(1)
        c3.metric("Most common Overall Rank", top.index[0] if len(top) else "—")
    else:
        c3.metric("Most common Overall Rank", "—")

    if ai_col and ai_col in df.columns:
        ai_rate = float(df[ai_col].fillna(False).mean() * 100)
        c4.metric("AI-Suspected Rate", f"{ai_rate:.1f}%")
    else:
        c4.metric("AI-Suspected Rate", "—")

    st.divider()

    # Donut / pie: Overall Rank distribution
    left, right = st.columns(2)

    if overall_rank_col and overall_rank_col in df.columns:
        rank_counts = df[overall_rank_col].replace({"": np.nan}).dropna().astype(str).value_counts().reset_index()
        rank_counts.columns = ["Overall Rank", "Count"]
        fig_pie = px.pie(rank_counts, names="Overall Rank", values="Count", hole=0.55)
        left.plotly_chart(fig_pie, use_container_width=True)
    else:
        left.info("Overall Rank column not found in this worksheet.")

    # AI donut (Suspected vs Not)
    if ai_col and ai_col in df.columns:
        tmp = pd.DataFrame({
            "AI": np.where(df[ai_col].fillna(False), "AI-Suspected", "Not Suspected")
        })
        ai_counts = tmp["AI"].value_counts().reset_index()
        ai_counts.columns = ["AI", "Count"]
        fig_ai = px.pie(ai_counts, names="AI", values="Count", hole=0.55)
        right.plotly_chart(fig_ai, use_container_width=True)
    else:
        right.info("AI flag column not found (AI_Suspected or AI-Suspected).")

    st.divider()

    # Histogram: Overall Total
    if overall_total_col and overall_total_col in df.columns:
        s = pd.to_numeric(df[overall_total_col], errors="coerce").dropna()
        if len(s):
            fig_hist = px.histogram(s, nbins=12)
            fig_hist.update_layout(xaxis_title=overall_total_col, yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Overall Total column exists but has no numeric values.")
    else:
        st.info("Overall Total column not found in this worksheet.")

    # AI_MaxScore (if present)
    if ai_maxscore_col and ai_maxscore_col in df.columns:
        st.divider()
        st.markdown("**AI_MaxScore distribution**")
        am = pd.to_numeric(df[ai_maxscore_col], errors="coerce").dropna()
        if len(am):
            fig_am = px.histogram(am, nbins=12)
            fig_am.update_layout(xaxis_title="AI_MaxScore", yaxis_title="Count")
            st.plotly_chart(fig_am, use_container_width=True)
        else:
            st.info("AI_MaxScore exists but has no numeric values.")

    st.divider()

    # Heatmap: attributes vs mean score (uses *_Avg (0–3) when available, else mean of Qn columns)
    attr_means = []
    for a, meta in attrs.items():
        val = None
        if meta["avg"] and meta["avg"] in df.columns:
            val = pd.to_numeric(df[meta["avg"]], errors="coerce").mean()
        else:
            cols = [meta["score_cols"][qn] for qn in meta["qns"] if qn in meta["score_cols"] and meta["score_cols"][qn] in df.columns]
            if cols:
                val = df[cols].apply(pd.to_numeric, errors="coerce").stack().mean()
        if val is not None and not np.isnan(val):
            attr_means.append({"Attribute": a, "MeanScore": float(val)})

    if attr_means:
        hm = pd.DataFrame(attr_means).sort_values("MeanScore", ascending=False)
        hm_mat = hm.set_index("Attribute")[["MeanScore"]]
        fig_attr_hm = px.imshow(hm_mat, text_auto=True, aspect="auto", labels=dict(color="Mean (0–3)"))
        st.markdown("**Heatmap: attribute mean scores (0–3)**")
        st.plotly_chart(fig_attr_hm, use_container_width=True)

        st.markdown("**Top / bottom attributes**")
        top3 = hm.head(3)[["Attribute","MeanScore"]]
        bot3 = hm.tail(3)[["Attribute","MeanScore"]]
        cL, cR = st.columns(2)
        cL.dataframe(top3, use_container_width=True, hide_index=True)
        cR.dataframe(bot3, use_container_width=True, hide_index=True)
    else:
        st.info("Could not compute attribute means (no Avg columns and no Qn columns detected).")

# -----------------------------
# Footer: preview (optional)
# -----------------------------
with st.expander("Preview data (first 20 rows)", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
