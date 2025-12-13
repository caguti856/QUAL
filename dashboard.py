# dashboard.py
# ------------------------------------------------------------
# PowerBI-like Dashboard (5 pages) reading scored data directly
# from Google Sheets worksheets using service account secrets.
# ------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

# ==============================
# CONFIG / SHEETS
# ==============================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

GSHEETS_SPREADSHEET_KEY = st.secrets.get("GSHEETS_SPREADSHEET_KEY", "")

GSHEETS_WORKSHEET_NAME  = "Advisory"
GSHEETS_WORKSHEET_NAME1 = "Thought Leadership"
GSHEETS_WORKSHEET_NAME2 = "Growth Mindset"
GSHEETS_WORKSHEET_NAME3 = "Networking"
GSHEETS_WORKSHEET_NAME4 = "Influencingrelationship"

PAGES = {
    "Thought Leadership": GSHEETS_WORKSHEET_NAME1,
    "Growth Mindset": GSHEETS_WORKSHEET_NAME2,
    "Networking & Advocacy": GSHEETS_WORKSHEET_NAME3,
    "Advisory Skills": GSHEETS_WORKSHEET_NAME,
    "Influencing Relationship": GSHEETS_WORKSHEET_NAME4,
}

# Short titles only (NO QUESTIONS)
THOUGHT_LEADERSHIP_TITLES = {
    ("Locally Anchored Visioning", 1): "Vision with Roots",
    ("Locally Anchored Visioning", 2): "Hard-wire Local Leadership",
    ("Locally Anchored Visioning", 3): "Safeguard Community Voice",
    ("Locally Anchored Visioning", 4): "Trade-off for Locally Led Scale",

    ("Innovation and Insight", 1): "Field-First Learning Loop",
    ("Innovation and Insight", 2): "Contradict HQ Assumptions",
    ("Innovation and Insight", 3): "Avoid Pilot Theatre",
    ("Innovation and Insight", 4): "Frugal Innovation Test",

    ("Execution Planning", 1): "Execution Spine Ownership",
    ("Execution Planning", 2): "90-Day Plan on a Page",
    ("Execution Planning", 3): "Prevent Handoff Failure",
    ("Execution Planning", 4): "Drop to Regain Clarity",

    ("Cross-Functional Collaboration", 1): "One-Day Alignment Workshop",
    ("Cross-Functional Collaboration", 2): "Resolve MEAL vs Gender Tension",
    ("Cross-Functional Collaboration", 3): "Shared Principles & Adherence",
    ("Cross-Functional Collaboration", 4): "Co-Owned Decision Design",

    ("Follow-Through Discipline", 1): "Executable Promise",
    ("Follow-Through Discipline", 2): "Light Dashboard & Escalation",
    ("Follow-Through Discipline", 3): "Milestone Recovery Options",
    ("Follow-Through Discipline", 4): "Stop Update Theatre",

    ("Learning-Driven Adjustment", 1): "Quarterly Pause-and-Reflect",
    ("Learning-Driven Adjustment", 2): "Hypothesis & Evidence Shift",
    ("Learning-Driven Adjustment", 3): "Handle Negative Findings",
    ("Learning-Driven Adjustment", 4): "Stop to Fund Adaptation",

    ("Result-Oriented Decision-Making", 1): "Ministry Alignment Trade-off",
    ("Result-Oriented Decision-Making", 2): "Decide-by-Friday Data",
    ("Result-Oriented Decision-Making", 3): "Socialize Hard Trade-off",
    ("Result-Oriented Decision-Making", 4): "Coherence Decision Rule",
}

# ==============================
# CSS (PowerBI-like feel)
# ==============================
def inject_css():
    st.markdown("""
    <style>
      :root{
        --primary:#F26A21;
        --bg:#0b1220;
        --card:#0f172a;
        --card2:#111b33;
        --text:#e5e7eb;
        --muted:#94a3b8;
        --border:rgba(148,163,184,0.18);
      }
      [data-testid="stAppViewContainer"]{
        background: radial-gradient(circle at top left, rgba(242,106,33,0.15) 0, rgba(15,23,42,1) 45%, rgba(2,6,23,1) 100%);
        color: var(--text);
      }
      [data-testid="stSidebar"]{
        background:#050a14;
        border-right:1px solid rgba(148,163,184,0.12);
      }
      [data-testid="stSidebar"] *{ color: var(--text) !important; }
      .block-container{ max-width: 1500px; padding-top: 1.1rem; padding-bottom: 3rem; }
      .pb-card{
        background: linear-gradient(180deg, rgba(17,27,51,0.95), rgba(15,23,42,0.95));
        border:1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.28);
      }
      .pb-title{ font-size: 1.6rem; font-weight: 800; margin: 0; }
      .pb-sub{ color: var(--muted); margin-top: 6px; font-size: 0.9rem; }
      .kpi{
        background: linear-gradient(135deg, rgba(242,106,33,0.22), rgba(17,27,51,0.95));
        border:1px solid var(--border);
        border-radius: 16px;
        padding: 12px 14px;
      }
      .kpi .label{ color: var(--muted); font-size: 0.8rem; }
      .kpi .value{ color: var(--text); font-size: 1.35rem; font-weight: 800; margin-top: 2px; }
      .kpi .hint{ color: var(--muted); font-size: 0.75rem; margin-top: 4px; }
      .divider{ height: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# GOOGLE SHEETS
# ==============================
def _normalize_sa_dict(raw: dict) -> dict:
    if not raw:
        raise ValueError("Missing st.secrets['gcp_service_account']")
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
        raise ValueError(f"Service account missing fields: {', '.join(missing)}")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets.get("gcp_service_account"))
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_data(ttl=120, show_spinner=False)
def read_worksheet_df(spreadsheet_key: str, worksheet_name: str) -> pd.DataFrame:
    if not spreadsheet_key:
        return pd.DataFrame()
    gc = gs_client()
    sh = gc.open_by_key(spreadsheet_key)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^\s*$")]
    return df

# ==============================
# HELPERS
# ==============================
def clean_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c or "")).strip()

def to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("true","1","yes","y","t")

def to_num_series(s: pd.Series):
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan, "None": np.nan}), errors="coerce")

QCOL_RX = re.compile(r"^(?P<section>.+)_Qn(?P<qn>[1-9]\d*)$", re.I)
RCOL_RX = re.compile(r"^(?P<section>.+)_Rubric_Qn(?P<qn>[1-9]\d*)$", re.I)
AVG_RX  = re.compile(r"^(?P<section>.+)_Avg\s*\(0[–-]3\)$", re.I)
RANK_RX = re.compile(r"^(?P<section>.+)_RANK$", re.I)

def discover_sections(df: pd.DataFrame):
    """
    Finds sections based on columns:
      Section_Qn1, Section_Rubric_Qn1, Section_Avg (0–3), Section_RANK
    """
    sections = {}
    cols = [clean_colname(c) for c in df.columns]

    for c in cols:
        m = QCOL_RX.match(c)
        if m:
            sec = m.group("section").strip()
            qn  = int(m.group("qn"))
            sections.setdefault(sec, {"qcols": {}, "rcols": {}, "avg_col": None, "rank_col": None})
            sections[sec]["qcols"][qn] = c

        m2 = RCOL_RX.match(c)
        if m2:
            sec = m2.group("section").strip()
            qn  = int(m2.group("qn"))
            sections.setdefault(sec, {"qcols": {}, "rcols": {}, "avg_col": None, "rank_col": None})
            sections[sec]["rcols"][qn] = c

        m3 = AVG_RX.match(c)
        if m3:
            sec = m3.group("section").strip()
            sections.setdefault(sec, {"qcols": {}, "rcols": {}, "avg_col": None, "rank_col": None})
            sections[sec]["avg_col"] = c

        m4 = RANK_RX.match(c)
        if m4:
            sec = m4.group("section").strip()
            sections.setdefault(sec, {"qcols": {}, "rcols": {}, "avg_col": None, "rank_col": None})
            sections[sec]["rank_col"] = c

    sections = {k:v for k,v in sections.items() if v.get("qcols")}
    for sec in sections:
        sections[sec]["qcols"] = dict(sorted(sections[sec]["qcols"].items()))
        sections[sec]["rcols"] = dict(sorted(sections[sec]["rcols"].items()))
    return dict(sorted(sections.items(), key=lambda kv: kv[0].lower()))

def find_overall_cols(df: pd.DataFrame):
    cols = {clean_colname(c).lower(): clean_colname(c) for c in df.columns}
    overall_total = None
    overall_rank  = None
    ai_flag       = None
    care_staff    = None

    for low, orig in cols.items():
        if care_staff is None and low in ("care_staff","care staff","staff id","staff_id"):
            care_staff = orig
        if overall_total is None and "overall total" in low:
            overall_total = orig
        if overall_rank is None and low == "overall rank":
            overall_rank = orig
        if ai_flag is None and ("ai_suspected" in low or "ai-suspected" in low or low == "ai suspected"):
            ai_flag = orig

    return care_staff, overall_total, overall_rank, ai_flag

def question_title(page_name: str, section: str, qn: int) -> str:
    if page_name == "Thought Leadership":
        t = THOUGHT_LEADERSHIP_TITLES.get((section, qn))
        if t:
            return t
    return f"Qn{qn}"

def score_pct_df(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    s = to_num_series(df[score_col])
    counts = s.value_counts(dropna=True).reindex([0,1,2,3], fill_value=0)
    total = int(counts.sum()) if int(counts.sum()) > 0 else 1
    pct = (counts / total * 100).round(1)
    return pd.DataFrame({"Score": [0,1,2,3], "Percent": pct.values, "Count": counts.values})

def rubric_freq_df(df: pd.DataFrame, rubric_col: str) -> pd.DataFrame:
    s = df[rubric_col].astype(str).str.strip().replace({"": np.nan, "None": np.nan})
    vc = s.value_counts(dropna=True)
    out = vc.reset_index()
    out.columns = ["Rubric", "Count"]
    return out

def heatmap_matrix(df: pd.DataFrame, qcols: dict, page_name: str, section: str) -> pd.DataFrame:
    rows = []
    for qn, col in qcols.items():
        title = question_title(page_name, section, qn)
        dist = score_pct_df(df, col).set_index("Score")["Percent"]
        rows.append({
            "Question": title,
            0: float(dist.loc[0]),
            1: float(dist.loc[1]),
            2: float(dist.loc[2]),
            3: float(dist.loc[3]),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Question")

# ==============================
# PAGE RENDER
# ==============================
def render_page(page_label: str, worksheet_name: str):
    with st.spinner(f"Loading: {worksheet_name} ..."):
        df_raw = read_worksheet_df(GSHEETS_SPREADSHEET_KEY, worksheet_name)

    if df_raw.empty:
        st.error(f"No data found in worksheet '{worksheet_name}'.")
        st.stop()

    df = df_raw.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    sections = discover_sections(df)
    care_staff_col, overall_total_col, overall_rank_col, ai_col = find_overall_cols(df)

    st.markdown(f"""
      <div class="pb-card">
        <div class="pb-title">{page_label}</div>
        <div class="pb-sub">Source worksheet: <b>{worksheet_name}</b> • Choose a section to view charts and heatmaps.</div>
      </div>
      <div class="divider"></div>
    """, unsafe_allow_html=True)

    # KPI row
    n = len(df)
    ai_rate = None
    if ai_col and ai_col in df.columns:
        ai_rate = round(df[ai_col].apply(to_bool).mean() * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
          <div class="kpi"><div class="label">Respondents</div><div class="value">{n:,}</div>
          <div class="hint">Rows in worksheet</div></div>
        """, unsafe_allow_html=True)

    with k2:
        if overall_total_col:
            ot = to_num_series(df[overall_total_col])
            st.markdown(f"""
              <div class="kpi"><div class="label">Avg Overall Total</div><div class="value">{np.nanmean(ot):.1f}</div>
              <div class="hint">{overall_total_col}</div></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='kpi'><div class='label'>Avg Overall Total</div><div class='value'>—</div></div>", unsafe_allow_html=True)

    with k3:
        if overall_rank_col:
            top = df[overall_rank_col].astype(str).str.strip().replace({"": np.nan}).value_counts().head(1)
            top_label = top.index[0] if len(top) else "—"
            st.markdown(f"""
              <div class="kpi"><div class="label">Most common rank</div><div class="value">{top_label}</div>
              <div class="hint">{int(top.iloc[0]) if len(top) else 0} respondents</div></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='kpi'><div class='label'>Most common rank</div><div class='value'>—</div></div>", unsafe_allow_html=True)

    with k4:
        if ai_rate is not None:
            st.markdown(f"""
              <div class="kpi"><div class="label">AI suspected</div><div class="value">{ai_rate:.1f}%</div>
              <div class="hint">{ai_col}</div></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='kpi'><div class='label'>AI suspected</div><div class='value'>—</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not sections:
        st.warning("No section question columns found (expected columns like 'Section_Qn1').")
        st.dataframe(df.head(30), use_container_width=True)
        st.stop()

    sec_names = list(sections.keys())
    section = st.selectbox("Section", sec_names, index=0, key=f"sec_{worksheet_name}")

    qcols = sections[section]["qcols"]
    rcols = sections[section]["rcols"]
    avg_col  = sections[section].get("avg_col")
    rank_col = sections[section].get("rank_col")

    # Section-level summary row (Avg + Rank)
    if avg_col or rank_col:
        c1, c2 = st.columns(2)
        with c1:
            if avg_col and avg_col in df.columns:
                av = to_num_series(df[avg_col])
                fig = px.histogram(av.dropna(), nbins=10, title=f"{section} — Avg (0–3) distribution")
                fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=320)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if rank_col and rank_col in df.columns:
                rk = df[rank_col].astype(str).str.strip().replace({"": np.nan}).value_counts(dropna=True).reset_index()
                rk.columns = ["Rank", "Count"]
                fig = px.bar(rk, x="Count", y="Rank", orientation="h", title=f"{section} — Rank frequency")
                fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=320)
                st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    mat = heatmap_matrix(df, qcols, page_label, section)
    if not mat.empty:
        fig_h = px.imshow(
            mat.values,
            x=[0,1,2,3],
            y=mat.index.tolist(),
            aspect="auto",
            labels=dict(x="Score", y="Question", color="%"),
        )
        fig_h.update_layout(
            title=f"{section} — Score distribution heatmap (%)",
            margin=dict(l=10, r=10, t=55, b=10),
            height=320 + 18*len(mat.index),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Per-question tiles (2 per row)
    st.markdown(f"<div class='pb-card'><b>{section}</b> — Question breakdown</div>", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    qnums = list(qcols.keys())
    for j in range(0, len(qnums), 2):
        row = st.columns(2)
        for k, qn in enumerate(qnums[j:j+2]):
            with row[k]:
                score_col = qcols[qn]
                rubric_col = rcols.get(qn)

                title = question_title(page_label, section, qn)

                # % distribution of scores 0..3 (column chart)
                dist = score_pct_df(df, score_col)
                fig1 = px.bar(
                    dist, x="Score", y="Percent", text="Percent",
                    title=f"{title} — Scores (0–3) %",
                )
                fig1.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
                fig1.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=320)
                st.plotly_chart(fig1, use_container_width=True)

                # rubric frequency (bar chart)
                if rubric_col and rubric_col in df.columns:
                    rf = rubric_freq_df(df, rubric_col)
                    fig2 = px.bar(rf, x="Count", y="Rubric", orientation="h", title=f"{title} — Rubric frequency")
                    fig2.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=320)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Rubric column not found for this question.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Overall (no dates/duration/AI_MaxScore)
    st.markdown("<div class='pb-card'><b>Overall</b></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        if overall_rank_col and overall_rank_col in df.columns:
            rk = df[overall_rank_col].astype(str).str.strip().replace({"": np.nan}).value_counts(dropna=True).reset_index()
            rk.columns = ["Overall Rank", "Count"]
            fig = px.bar(rk, x="Count", y="Overall Rank", orientation="h", title="Overall Rank — frequency")
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if overall_total_col and overall_total_col in df.columns:
            ot = to_num_series(df[overall_total_col])
            fig = px.histogram(ot.dropna(), nbins=12, title=f"{overall_total_col} — distribution")
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

    if ai_col and ai_col in df.columns:
        ai = df[ai_col].apply(to_bool)
        pie = pd.DataFrame({"AI_Suspected": ["TRUE","FALSE"], "Count": [int(ai.sum()), int((~ai).sum())]})
        fig = px.pie(pie, values="Count", names="AI_Suspected", title="AI_Suspected — share")
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Optional preview (keeps it clean)
    with st.expander("Preview data (first 20 rows)"):
        hide = {"date","duration_min","ai_maxscore","ai_score_max"}
        cols_show = [c for c in df.columns if c.lower() not in hide]
        st.dataframe(df[cols_show].head(20), use_container_width=True)

# ==============================
# MAIN
# ==============================
def main():
    st.set_page_config(page_title="Scoring Dashboard", layout="wide")
    inject_css()

    st.sidebar.markdown("## Dashboard Pages")
    page = st.sidebar.radio("Select page", list(PAGES.keys()), index=0)

    if not GSHEETS_SPREADSHEET_KEY:
        st.error("Missing GSHEETS_SPREADSHEET_KEY in st.secrets.")
        st.stop()

    render_page(page, PAGES[page])

if __name__ == "__main__":
    main()
