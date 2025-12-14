# dashboard.py
# ------------------------------------------------------------
# Updates (locked to your color codes + your layout rules):
# - Keeps your exact theme colors: APP_BG, CARD_BG, sidebar purple, chart bg, BLUES + HEAT_SCALE
# - Adds CARE Staff ID filter in the sidebar (optional; auto-detects common column names)
# - Fixes "tab spill/duplication": EVERYTHING for a section stays inside that section tab
# - Avg = chart ; Rank = table (per your rule)
# - Score = chart + table ; Rubric = chart + full table in expander
# - Heatmap shown once per section
# - Unique keys for all plotly charts (avoids StreamlitDuplicateElementId)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Scoring Dashboard", layout="wide")

# ==============================
# YOUR THEME / COLORS (kept)
# ==============================
APP_BG = "#E5C7AB"       # your "grey" background choice
CARD_BG = "#EB7100"      # your card color choice
BORDER = "rgba(0,0,0,0.12)"
CT ="#D8D5E9"
TEXT = "#FFFFFF"
MUTED = "rgba(0,0,0,0.70)"
BLA="#090015"
CHART_PAPER = "#313132"
CHART_PLOT  = "#D7D6D4"
METRIC_LABEL_SIZE = 26
METRIC_VALUE_SIZE = 22
METRIC_LABEL_COLOR = "#090015"   # example: your BLA (dark)
METRIC_VALUE_COLOR = "#EB7100"   # example: your TEXT

BLUES = ["#031432", "#0B47A8", "#1D7AFC", "#3B8EF5", "#74A8FF", "#073072"]
HEAT_SCALE = [
    [0.00, "#073072"],
    [0.25, "#E2D6FF"],
    [0.55, "#1D7AFC"],
    [0.80, "#0B47A8"],
    [1.00, "#031432"],
]

def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {APP_BG};
            color: {TEXT};
        }}
        .block-container {{
            padding-top: 1.1rem;
        }}

        /* Global text */
        html, body, [class*="css"] {{
            color: {TEXT} !important;
            font-size: 20px !important;
        }}

        /* Headings */
        h1 {{
            font-size: 32px !important;
            font-weight: 900 !important;
            color: {TEXT} !important;
            letter-spacing: .2px;
        }}
        h2 {{
            font-size: 28px !important;
            font-weight: 900 !important;
            color: {TEXT} !important;
        }}
        h3 {{
            font-size: 20px !important;
            font-weight: 900 !important;
            color: {TEXT} !important;
        }}
        h4 {{
            font-size: 20px !important;
            font-weight: 900 !important;
            color: {TEXT} !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: #241E4E;
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}

        /* Cards */
        .card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.06);
        }}
        .card-title {{
            font-size: 28px !important;
            letter-spacing: .45px;
            text-transform: uppercase;
            font-weight: 900;
            color: {TEXT} !important;
            margin-bottom: 6px;
            opacity: .85;
        }}
        .card-value {{
            font-size: 26px !important;
            font-weight: 900;
            color: {TEXT} !important;
            line-height: 1.05;
        }}
        .card-sub {{
            font-size: 24px !important;
            color: {TEXT} !important;
            opacity: .85;
            margin-top: 6px;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            font-weight: 900 !important;
            color: {BLA} !important;
            font-size: 20px !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-bottom: 4px solid {BLUES[0]} !important;
        }}

        /* Dataframes */
        .stDataFrame {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {BORDER};
            background: {CARD_BG};
        }}

        /* Metric label (Mean Total etc.) */
        div[data-testid="stMetricLabel"] > div {{
            font-size: {METRIC_LABEL_SIZE}px !important;
            color: {METRIC_LABEL_COLOR} !important;
            font-weight: 800 !important;
        }}

        /* Metric value (13.4 etc.) */
        div[data-testid="stMetricValue"] > div {{
            font-size: {METRIC_VALUE_SIZE}px !important;
            color: {METRIC_VALUE_COLOR} !important;
            font-weight: 900 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



def card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================
# CONFIG (Worksheets)
# ==============================
WS_THOUGHT   = "Thought Leadership"
WS_GROWTH    = "Growth Mindset"
WS_NETWORK   = "Networking"
WS_ADVISORY  = "Advisory"
WS_INFLUENCE = "Influencingrelationship"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SPREADSHEET_KEY = st.secrets["GSHEETS_SPREADSHEET_KEY"]

# ==============================
# SECTION DEFINITIONS
# ==============================
PAGES = {
    "Thought Leadership": {
        "worksheet": WS_THOUGHT,
        "sections": [
            "Locally Anchored Visioning",
            "Innovation and Insight",
            "Execution Planning",
            "Cross-Functional Collaboration",
            "Follow-Through Discipline",
            "Learning-Driven Adjustment",
            "Result-Oriented Decision-Making",
        ],
        "overall_total_col": "Overall Total (0–21)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {
            "Locally Anchored Visioning": [
                "Vision with Roots",
                "Local Leadership in ToRs/Budgets",
                "Safeguard Community Voice",
                "Trade Ambition vs Protection",
            ],
            "Innovation and Insight": [
                "Field-First Learning Loop",
                "Surface Contradicting Insights",
                "Avoid Pilot Theatre",
                "Frugal Innovation Test",
            ],
            "Execution Planning": [
                "Execution Spine Ownership",
                "90-Day Plan & Decision Gates",
                "Close Handoff Failure",
                "Drop to Regain Clarity",
            ],
            "Cross-Functional Collaboration": [
                "One-Day Alignment Workshop",
                "Resolve MEAL vs Gender Tension",
                "Shared Principles & Tests",
                "Co-Owned Decision Design",
            ],
            "Follow-Through Discipline": [
                "Executable Promise",
                "Light Dashboard & Escalation",
                "Recovery Conversation",
                "Stop Update Theatre",
            ],
            "Learning-Driven Adjustment": [
                "Quarterly Pause & Reflect",
                "Strategic Hypothesis & Evidence",
                "Handle Negative Findings",
                "Stop to Fund Adaptations",
            ],
            "Result-Oriented Decision-Making": [
                "Agro-Parks vs Grassroots Call",
                "Decision by Friday",
                "Socialize Hard Trade-Off",
                "Decision Rule for Divergence",
            ],
        },
    },

    "Growth Mindset": {
        "worksheet": WS_GROWTH,
        "sections": ["Learning Agility", "Digital Savvy", "Innovation", "Contextual Intelligence"],
        "overall_total_col": "Overall Total (0–12)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {
            "Learning Agility": [
                "Correct & Improve Quickly",
                "Test Ideas & Change Mind",
                "Listen to Rumours as Signals",
                "Co-Design in Market Cycle",
            ],
            "Digital Savvy": [
                "USSD vs Offline vs Smartphone",
                "Consent for Low Literacy",
                "Offline Transaction Workflow",
                "Correct Mistakes Transparently",
            ],
            "Innovation": [
                "Prototype A vs B Test Design",
                "Low-Cost Inclusion Idea",
                "Co-Design Session Outputs",
                "Public ‘We Heard You’ Message",
            ],
            "Contextual Intelligence": [
                "Actor Map & Red Lines",
                "Handle ‘Facilitation’ Pressure",
                "Safeguards for Fee Link",
                "Radio Trust Rebuild Plan",
            ],
        },
    },

    "Networking & Advocacy": {
        "worksheet": WS_NETWORK,
        "sections": [
            "Strategic Positioning & Donor Fluency",
            "Power-Aware Stakeholder Mapping",
            "Equitable Allyship & Local Fronting",
            "Coalition Governance & Convening",
            "Community-Centered Messaging",
            "Evidence-Led Learning (Local Knowledge)",
            "Influence Without Authority",
            "Risk Management & Adaptive Communication",
        ],
        "overall_total_col": "Overall Total (0–24)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},
    },

    "Advisory Skills": {
        "worksheet": WS_ADVISORY,
        "sections": [
            "Strategic & analytical thinking",
            "Credibility & trustworthiness",
            "Effective communication & influence",
            "Client & stakeholder focus",
            "Fostering collaboration & partnership",
            "Ensuring relevance & impact",
            "Solution orientation & adaptability",
            "Capacity strengthening & empowerment support",
        ],
        "overall_total_col": "Overall Total (0–24)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},
    },

    "Influencing Relationships": {
        "worksheet": WS_INFLUENCE,
        "sections": [
            "Strategic Positioning & Political Acumen",
            "Stakeholder Mapping & Engagement",
            "Evidence-Informed Advocacy",
            "Communication, Framing & Messaging",
            "Risk Awareness & Mitigation",
            "Coalition Building & Collaborative Action",
            "Adaptive Tactics & Channel Selection",
            "Integrity & Values-Based Influencing",
        ],
        "overall_total_col": "Overall Total (0–24)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},
    },
}

# ==============================
# GOOGLE SHEETS
# ==============================
def _normalize_sa_dict(raw: dict) -> dict:
    sa = dict(raw)
    if sa.get("private_key") and "\\n" in sa["private_key"]:
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    sa.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner=False)
def load_sheet_df(sheet_name: str) -> pd.DataFrame:
    gc = gs_client()
    ws = gc.open_by_key(SPREADSHEET_KEY).worksheet(sheet_name)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ==============================
# UTIL / CLEANING
# ==============================
def drop_date_duration_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    bad = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if "duration" in cl or "date" in cl:
            bad.append(c)
    return df.drop(columns=bad, errors="ignore")

def _clean_series(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()

def has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c and c in df.columns:
            return c
    return None

def safe_plot(fig, key: str):
    fig.update_layout(
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PLOT,
        font=dict(color=TEXT, size=16),
        title_font=dict(size=20, color=TEXT, family="Arial Black"),
        legend_font=dict(color=TEXT, size=14),
        margin=dict(l=10, r=10, t=70, b=15),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)",
        zeroline=False,
        tickfont=dict(color=TEXT, size=14),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)",
        zeroline=False,
        tickfont=dict(color=TEXT, size=14),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ==============================
# STAFF ID FILTER (optional)
# ==============================
def staff_id_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Care_Staff",
        "CARE Staff id",
        "CARE Staff ID",
        "CARE_Staff_ID",
        "Staff ID",
        "StaffID",
        "Staff Id",
        "CAREStaffID",
    ]
    return first_existing(df, candidates)

# ==============================
# TABLE + METRICS
# ==============================
def top_value(series: pd.Series) -> tuple[str, int]:
    s = _clean_series(series)
    if not len(s):
        return ("-", 0)
    vc = s.value_counts()
    return (str(vc.index[0]), int(vc.iloc[0]))

def score_dist(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not has(df, col):
        return pd.DataFrame({"Score": [0, 1, 2, 3], "Percent": [0, 0, 0, 0], "Count": [0, 0, 0, 0]})
    s = pd.to_numeric(df[col], errors="coerce")
    counts = s.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100).round(1) if total else counts.astype(float)
    out = pd.DataFrame({"Score": [0, 1, 2, 3], "Count": counts.values, "Percent": pct.values})
    return out

def rubric_freq(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not has(df, col):
        return pd.DataFrame(columns=["Rubric", "Count"])
    x = _clean_series(df[col])
    out = x.value_counts().reset_index()
    out.columns = ["Rubric", "Count"]
    return out

def section_heatmap(df: pd.DataFrame, section: str, q_titles: list[str]) -> pd.DataFrame:
    rows = []
    for qn in range(1, 5):
        col = f"{section}_Qn{qn}"
        if not has(df, col):
            continue
        title = q_titles[qn - 1] if (q_titles and len(q_titles) >= qn) else f"Qn{qn}"
        d = score_dist(df, col)
        rows.append({
            "Question": title,
            "0": float(d.loc[d["Score"] == 0, "Percent"].values[0]),
            "1": float(d.loc[d["Score"] == 1, "Percent"].values[0]),
            "2": float(d.loc[d["Score"] == 2, "Percent"].values[0]),
            "3": float(d.loc[d["Score"] == 3, "Percent"].values[0]),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Question")

# ==============================
# RENDER ONE PAGE
# ==============================
def render_page(page_name: str):
    cfg = PAGES[page_name]
    ws_name = cfg["worksheet"]

    with st.spinner(f"Loading: {ws_name} ..."):
        df = load_sheet_df(ws_name)
    df = drop_date_duration_cols(df)

    # Sidebar filters (page-scoped)
    
    with st.sidebar:
        st.markdown("## Filters")

    sid_col = staff_id_column(df)
    selected_staff = ["All"]  # multiselect returns a LIST

    if sid_col:
        staff_ids = (
            df[sid_col].astype(str)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            .dropna()
            .unique()
        )
        staff_ids = sorted(staff_ids, key=lambda x: (len(str(x)), str(x)))

        options = ["All"] + list(staff_ids)

        with st.sidebar:
            selected_staff = st.multiselect(
                "CARE Staff ID",
                options=options,
                default=["All"],   # ✅ correct for multiselect
                key=f"{page_name}-staff-filter",
            )

        # ---- filtering logic (robust)
        if not selected_staff or "All" in selected_staff:
            # no filtering if none selected or "All" chosen
            pass
        else:
            df = df[df[sid_col].astype(str).isin([str(x) for x in selected_staff])]
    else:
        with st.sidebar:
            st.caption("No Staff ID column found on this sheet.")


    # Header cards (only 2, balanced)
    cA, cB = st.columns(2)
    with cA:
        sub = f"Staff: {selected_staff}" if selected_staff != "All" else ""
        card("SECTION", ws_name, sub)
    with cB:
        card("Total Responses", f"{len(df):,}")

    st.write("")
    tabs = st.tabs(cfg["sections"] + ["SUMMARY"])

    # ==========================
    # SECTION TABS (NO SPILL)
    # ==========================
    for i, section in enumerate(cfg["sections"]):
        with tabs[i]:
            st.subheader(section)

            avg_col = f"{section}_Avg (0–3)"
            rnk_col = f"{section}_RANK"

            # Avg = chart ; Rank = table
            t1, t2 = st.columns(2)

            with t1:
                st.markdown("#### Section Average Distribution")
                if has(df, avg_col):
                    s = pd.to_numeric(df[avg_col], errors="coerce").dropna()
                    if len(s):
                        avg_df = (
                            s.round(2)
                             .value_counts()
                             .sort_index()
                             .reset_index()
                        )
                        avg_df.columns = ["Avg (0–3)", "Count"]

                        fig = px.bar(
                            avg_df,
                            x="Avg (0–3)",
                            y="Count",
                            title="Average Score Distribution",
                            color_discrete_sequence=[BLUES[2]],
                        )
                        fig.update_layout(xaxis_title="Average (0–3)", yaxis_title="Count")
                        safe_plot(fig, key=f"{page_name}-{section}-avg-chart")
                    else:
                        st.info("No average values.")
                else:
                    st.info("Average column missing.")

            with t2:
                st.markdown("#### Section Rank Distribution (Table)")
                if has(df, rnk_col):
                    r = _clean_series(df[rnk_col])
                    if len(r):
                        r_tbl = r.value_counts().reset_index()
                        r_tbl.columns = ["Rank", "Count"]
                        r_tbl["Percent"] = (r_tbl["Count"] / r_tbl["Count"].sum() * 100).round(1)
                        st.dataframe(r_tbl, use_container_width=True, hide_index=True)
                    else:
                        st.info("No rank values.")
                else:
                    st.info("Rank column missing.")

            st.divider()

            # Qn blocks (ONLY inside this section tab)
            q_titles = cfg.get("question_titles", {}).get(section, [])

            for qn in range(1, 5):
                score_col  = f"{section}_Qn{qn}"
                rubric_col = f"{section}_Rubric_Qn{qn}"

                if (not has(df, score_col)) and (not has(df, rubric_col)):
                    continue

                title = q_titles[qn - 1] if (q_titles and len(q_titles) >= qn) else f"Qn{qn}"
                st.markdown(f"### {title}")

                left, right = st.columns(2)

                with left:
                    if has(df, score_col):
                        d = score_dist(df, score_col)
                        fig = px.bar(
                            d,
                            x="Score",
                            y="Percent",
                            text="Percent",
                            title="Score Distribution (%)",
                            color="Score",
                            color_discrete_sequence=BLUES,
                        )
                        fig.update_traces(texttemplate="%{text}%", textposition="outside")
                        fig.update_layout(yaxis_title="Percent", xaxis_title="Score (0–3)")
                        safe_plot(fig, key=f"{page_name}-{section}-qn{qn}-score")

                        st.markdown("**Score Table**")
                        st.dataframe(d[["Score", "Count", "Percent"]], use_container_width=True, hide_index=True)
                    else:
                        st.info("Score column missing.")

                with right:
                    if has(df, rubric_col):
                        rf = rubric_freq(df, rubric_col)
                        if len(rf):
                            fig = px.bar(
                                rf.head(12),
                                x="Count",
                                y="Rubric",
                                orientation="h",
                                title="Rubric Frequency (Top 12)",
                                color_discrete_sequence=[BLUES[1]],
                            )
                            safe_plot(fig, key=f"{page_name}-{section}-qn{qn}-rubricbar")

                            with st.expander("Full rubric table"):
                                st.dataframe(rf, use_container_width=True, hide_index=True)
                        else:
                            st.info("Rubric has no values.")
                    else:
                        st.info("Rubric column missing.")

                st.divider()

            # Heatmap ONCE per section
            st.markdown("### Heatmap (Score % by Question)")
            try:
                h = section_heatmap(df, section, q_titles)
                if not h.empty:
                    fig = px.imshow(
                        h,
                        labels=dict(x="Score", y="Question", color="%"),
                        title=f"{section} Heatmap",
                        color_continuous_scale=HEAT_SCALE,
                    )
                    safe_plot(fig, key=f"{page_name}-{section}-heatmap")
                else:
                    st.info("Heatmap skipped (missing Qn columns or values).")
            except Exception:
                st.info("Heatmap skipped.")

    # ==========================
    # OVERALL TAB
    # ==========================
    with tabs[-1]:
        st.subheader("Overall")

        overall_total = cfg["overall_total_col"]
        overall_rank  = cfg["overall_rank_col"]
        ai_col = first_existing(df, [cfg.get("ai_col"), "AI_Suspected", "AI-Suspected"])
        ai_maxscore_col = first_existing(df, [cfg.get("ai_maxscore_col"), "AI_MaxScore"])

        c1, c2, c3, c4 = st.columns(4)
        if has(df, overall_total):
            tot = pd.to_numeric(df[overall_total], errors="coerce")
            c1.metric("Mean Total", f"{tot.mean():.1f}" if tot.notna().any() else "-")
            c2.metric("Median Total", f"{tot.median():.1f}" if tot.notna().any() else "-")
        else:
            c1.metric("Mean Total", "-")
            c2.metric("Median Total", "-")

        if has(df, overall_rank):
            top_r, top_n = top_value(df[overall_rank])
            c3.metric("Top Overall Rank", top_r)
            c4.metric("Top Rank Count", f"{top_n:,}")
        else:
            c3.metric("Top Overall Rank", "-")
            c4.metric("Top Rank Count", "-")

        st.divider()

        if has(df, overall_total):
            fig = px.histogram(
                df, x=overall_total, title="Overall Total Distribution",
                color_discrete_sequence=[BLUES[2]]
            )
            fig.update_layout(xaxis_title="Overall Total", yaxis_title="Count")
            safe_plot(fig, key=f"{page_name}-overall-totaldist")

            t = pd.to_numeric(df[overall_total], errors="coerce").dropna()
            if len(t):
                summary = pd.DataFrame([{
                    "Min": float(t.min()),
                    "Max": float(t.max()),
                    "Mean": float(t.mean()),
                    "Median": float(t.median()),
                    "N": int(len(t)),
                }])
                st.markdown("**Overall Total Summary**")
                st.dataframe(summary, use_container_width=True, hide_index=True)

        if has(df, overall_rank):
            r = _clean_series(df[overall_rank])
            if len(r):
                r_df = r.value_counts().reset_index()
                r_df.columns = ["Rank", "Count"]
                fig = px.bar(
                    r_df, x="Rank", y="Count", title="Overall Rank Distribution",
                    color="Rank", color_discrete_sequence=BLUES
                )
                fig.update_layout(xaxis_title="", yaxis_title="Count")
                safe_plot(fig, key=f"{page_name}-overall-rankbar")

                st.markdown("**Overall Rank Table**")
                st.dataframe(r_df, use_container_width=True, hide_index=True)

        st.divider()

        if ai_col and has(df, ai_col):
            ai = _clean_series(df[ai_col])
            if len(ai):
                ai_df = ai.value_counts().reset_index()
                ai_df.columns = ["AI_Flag", "Count"]

                if len(ai_df) == 2:
                    left, right = st.columns(2)
                    with left:
                        fig = px.pie(
                            ai_df, names="AI_Flag", values="Count", hole=0.55,
                            title="AI Flag (Donut)",
                            color_discrete_sequence=[BLUES[2], BLUES[0]]
                        )
                        safe_plot(fig, key=f"{page_name}-overall-ai-donut")
                    with right:
                        st.markdown("**AI Flag Table**")
                        st.dataframe(ai_df, use_container_width=True, hide_index=True)
                else:
                    fig = px.bar(
                        ai_df, x="AI_Flag", y="Count", title="AI Flag Distribution",
                        color="AI_Flag", color_discrete_sequence=BLUES
                    )
                    fig.update_layout(xaxis_title="", yaxis_title="Count")
                    safe_plot(fig, key=f"{page_name}-overall-ai-bar")

                    st.markdown("**AI Flag Table**")
                    st.dataframe(ai_df, use_container_width=True, hide_index=True)

        if ai_maxscore_col and has(df, ai_maxscore_col):
            fig = px.histogram(
                df, x=ai_maxscore_col, title="AI_MaxScore Distribution",
                color_discrete_sequence=[BLUES[1]]
            )
            fig.update_layout(xaxis_title="AI_MaxScore", yaxis_title="Count")
            safe_plot(fig, key=f"{page_name}-overall-ai-maxscore")

# ==============================
# MAIN
# ==============================
def main():
    inject_css()
    st.title("ANALYTICS DASHBOARD")

    page = st.sidebar.radio("Dashboard Pages", list(PAGES.keys()), index=0)
    render_page(page)

if __name__ == "__main__":
    main()
