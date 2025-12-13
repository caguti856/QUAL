# dashboard.py
# ------------------------------------------------------------
# ONE Streamlit "PowerBI-style" dashboard with 5 PAGE-TABS:
# 1) Thought Leadership
# 2) Growth Mindset
# 3) Networking & Advocacy
# 4) Advisory Skills
# 5) Influencing Relationships
#
# Each page pulls its OWN worksheet from the SAME Google Sheet,
# then shows: section tabs (inside the page) + an Overall tab.
#
# Rules:
# - Pull from Google Sheets using gcp_service_account + GSHEETS_SPREADSHEET_KEY
# - DO NOT show questions (only short titles you define)
# - NO Date / Duration fields
# - Uses AI_Suspected (and AI_MaxScore where present)
# - Heatmaps + donut/pie (only when 2 cats) + bar/column + histograms
# - Avoid StreamlitDuplicateElementId by unique keys
# - Safe when data missing: skip quietly (no crash)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Scoring Dashboard", layout="wide")

# ==============================
# THEME / COLORS (Cute + Professional)
# ==============================
BRAND_ORANGE = "#EB7100"
BRAND_PURPLE = "#241E4E"
BRAND_CREAM  = "#FFF8EE"
BRAND_SOFT   = "#F9F6F4"

ACCENT_GOLD = "#D4AF37"
ACCENT_ROSE = "#FF5C8A"
ACCENT_TEAL = "#17BEBB"
ACCENT_SKY  = "#4EA8FF"
ACCENT_LIME = "#8FD14F"

DISCRETE_SEQ = [
    BRAND_ORANGE,
    BRAND_PURPLE,
    ACCENT_TEAL,
    ACCENT_SKY,
    ACCENT_ROSE,
    ACCENT_GOLD,
    ACCENT_LIME,
]

HEAT_SCALE = [
    [0.00, "#FFFFFF"],
    [0.20, BRAND_SOFT],
    [0.45, "#FFE3C8"],
    [0.70, BRAND_ORANGE],
    [1.00, BRAND_PURPLE],
]

def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at 15% 0%, #FFFFFF 0%, {BRAND_SOFT} 45%, {BRAND_CREAM} 100%);
        }}
        .block-container {{
            padding-top: 1.2rem;
        }}

        /* Shiny main title */
        h1 {{
            font-weight: 900 !important;
            letter-spacing: .6px !important;
            background: linear-gradient(90deg, {ACCENT_GOLD}, #FFD86B, {ACCENT_GOLD});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0px 10px 25px rgba(212,175,55,.18);
        }}
        h2, h3, h4 {{
            font-weight: 850 !important;
            color: {BRAND_PURPLE} !important;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, {BRAND_SOFT} 100%);
            border-right: 1px solid rgba(36, 30, 78, 0.10);
        }}

        .card {{
            background: linear-gradient(180deg, #FFFFFF 0%, {BRAND_SOFT} 100%);
            border: 1px solid rgba(36, 30, 78, 0.10);
            border-radius: 20px;
            padding: 16px 18px;
            box-shadow: 0 12px 30px rgba(36,30,78,0.10);
        }}
        .card-title {{
            font-size: 12px;
            letter-spacing: .4px;
            text-transform: uppercase;
            font-weight: 800;
            color: rgba(36,30,78,.70);
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 30px;
            font-weight: 900;
            color: {BRAND_PURPLE};
            line-height: 1.05;
        }}
        .card-sub {{
            font-size: 12px;
            color: rgba(36,30,78,.60);
            margin-top: 6px;
        }}

        button[data-baseweb="tab"] {{
            font-weight: 800 !important;
            color: rgba(36,30,78,.80) !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {BRAND_PURPLE} !important;
            border-bottom: 3px solid {BRAND_ORANGE} !important;
        }}

        .stDataFrame {{
            border-radius: 14px;
            overflow: hidden;
        }}

        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, {BRAND_SOFT} 100%);
            border: 1px solid rgba(36, 30, 78, 0.10);
            padding: 12px 14px;
            border-radius: 16px;
            box-shadow: 0 10px 24px rgba(36,30,78,0.08);
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
# SECTION DEFINITIONS (titles only — no questions)
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
        "ai_col": "AI_Suspected",
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
        "ai_col": "AI_Suspected",
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
        "ai_col": "AI_Suspected",
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
        "ai_col": "AI_Suspected",
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
        "ai_col": "AI_Suspected",
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
    """Unique key avoids StreamlitDuplicateElementId."""
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=20, family="Arial", color=BRAND_PURPLE),
        legend_title_text="",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ==============================
# METRICS + TABLE HELPERS
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
    return pd.DataFrame({"Score": [0, 1, 2, 3], "Percent": pct.values, "Count": counts.values})

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

    # Header cards
    cA, cB, cC = st.columns([1.2, 1.2, 1.6])
    with cA:
        card("Worksheet", ws_name, "Source: Google Sheets")
    with cB:
        card("Rows", f"{len(df):,}", "Total responses")
    with cC:
        card("Columns", f"{len(df.columns):,}", "Dates/duration removed")

    st.write("")

    section_tabs = cfg["sections"] + ["Overall"]
    tabs = st.tabs(section_tabs)

    # ======================
    # SECTION TABS
    # ======================
    for i, section in enumerate(cfg["sections"]):
        with tabs[i]:
            st.subheader(section)

            avg_col = f"{section}_Avg (0–3)"
            rnk_col = f"{section}_RANK"

            m1, m2, m3, m4 = st.columns(4)

            if has(df, avg_col):
                avg_vals = pd.to_numeric(df[avg_col], errors="coerce")
                m1.metric("Mean Avg", f"{avg_vals.mean():.2f}" if avg_vals.notna().any() else "-")
                m2.metric("Median Avg", f"{avg_vals.median():.2f}" if avg_vals.notna().any() else "-")
            else:
                m1.metric("Mean Avg", "-")
                m2.metric("Median Avg", "-")

            if has(df, rnk_col):
                top_rank, top_count = top_value(df[rnk_col])
                m3.metric("Top Rank", top_rank)
                m4.metric("Top Rank Count", f"{top_count:,}")
            else:
                m3.metric("Top Rank", "-")
                m4.metric("Top Rank Count", "-")

            st.divider()

            # Rank distribution (BAR only — multi category)
            if has(df, rnk_col):
                r = _clean_series(df[rnk_col])
                if len(r):
                    r_df = r.value_counts().reset_index()
                    r_df.columns = ["Rank", "Count"]
                    fig = px.bar(
                        r_df, x="Rank", y="Count",
                        title="Section Rank Distribution",
                        color="Rank", color_discrete_sequence=DISCRETE_SEQ
                    )
                    fig.update_layout(xaxis_title="", yaxis_title="Count")
                    safe_plot(fig, key=f"{page_name}-{section}-rankbar")
                else:
                    st.info("Section rank has no values (skipped).")

            # Avg distribution (histogram)
            if has(df, avg_col):
                fig = px.histogram(
                    df, x=avg_col, nbins=10,
                    title="Section Average Distribution",
                    color_discrete_sequence=[BRAND_ORANGE]
                )
                fig.update_layout(xaxis_title="Average (0–3)", yaxis_title="Count")
                safe_plot(fig, key=f"{page_name}-{section}-avgdist")

            st.divider()

            q_titles = cfg.get("question_titles", {}).get(section, [])

            for qn in range(1, 5):
                score_col  = f"{section}_Qn{qn}"
                rubric_col = f"{section}_Rubric_Qn{qn}"

                if (not has(df, score_col)) and (not has(df, rubric_col)):
                    continue

                title = q_titles[qn - 1] if (q_titles and len(q_titles) >= qn) else f"Qn{qn}"
                st.markdown(f"### {title}")

                left, right = st.columns(2)

                # Score distribution (column chart)
                with left:
                    if has(df, score_col):
                        d = score_dist(df, score_col)
                        fig = px.bar(
                            d, x="Score", y="Percent", text="Percent",
                            title="Score Distribution (%)",
                            color="Score",
                            color_discrete_sequence=DISCRETE_SEQ
                        )
                        fig.update_traces(texttemplate="%{text}%", textposition="outside")
                        fig.update_layout(yaxis_title="Percent", xaxis_title="Score (0–3)")
                        safe_plot(fig, key=f"{page_name}-{section}-qn{qn}-score")
                    else:
                        st.info("Score column missing (skipped).")

                # Rubric frequency (bar + table)
                with right:
                    if has(df, rubric_col):
                        rf = rubric_freq(df, rubric_col)
                        if len(rf):
                            fig = px.bar(
                                rf.head(12),
                                x="Count", y="Rubric", orientation="h",
                                title="Rubric Frequency (Top 12)",
                                color_discrete_sequence=[BRAND_PURPLE]
                            )
                            safe_plot(fig, key=f"{page_name}-{section}-qn{qn}-rubricbar")

                            with st.expander("See full rubric table"):
                                st.dataframe(rf, use_container_width=True)
                        else:
                            st.info("Rubric has no values (skipped).")
                    else:
                        st.info("Rubric column missing (skipped).")

                st.divider()

            # Heatmap (Score % by question)
            st.markdown("### Heatmap (Score % by Question)")
            try:
                h = section_heatmap(df, section, q_titles)
                if not h.empty:
                    fig = px.imshow(
                        h,
                        labels=dict(x="Score", y="Question", color="%"),
                        title=f"{section} Heatmap",
                        color_continuous_scale=HEAT_SCALE
                    )
                    safe_plot(fig, key=f"{page_name}-{section}-heatmap")
                else:
                    st.info("Heatmap skipped (missing Qn columns or no values).")
            except Exception:
                st.info("Heatmap skipped.")

    # ======================
    # OVERALL TAB
    # ======================
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

        # Overall Total distribution
        if has(df, overall_total):
            fig = px.histogram(
                df, x=overall_total,
                title="Overall Total Distribution",
                color_discrete_sequence=[BRAND_ORANGE]
            )
            fig.update_layout(xaxis_title="Overall Total", yaxis_title="Count")
            safe_plot(fig, key=f"{page_name}-overall-totaldist")
        else:
            st.info("Overall Total not available (skipped).")

        # Overall Rank distribution (bar)
        if has(df, overall_rank):
            r = _clean_series(df[overall_rank])
            if len(r):
                r_df = r.value_counts().reset_index()
                r_df.columns = ["Rank", "Count"]
                fig = px.bar(
                    r_df, x="Rank", y="Count",
                    title="Overall Rank Distribution",
                    color="Rank", color_discrete_sequence=DISCRETE_SEQ
                )
                fig.update_layout(xaxis_title="", yaxis_title="Count")
                safe_plot(fig, key=f"{page_name}-overall-rankbar")

                with st.expander("See Overall Rank table"):
                    st.dataframe(r_df, use_container_width=True)
            else:
                st.info("Overall Rank has no values (skipped).")
        else:
            st.info("Overall Rank not available (skipped).")

        st.divider()

        # AI flag: donut only when EXACTLY 2 categories, else bar
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
                            color_discrete_sequence=[BRAND_ORANGE, BRAND_PURPLE]
                        )
                        safe_plot(fig, key=f"{page_name}-overall-ai-donut")
                    with right:
                        st.dataframe(ai_df, use_container_width=True)
                else:
                    fig = px.bar(
                        ai_df, x="AI_Flag", y="Count",
                        title="AI Flag Distribution",
                        color="AI_Flag", color_discrete_sequence=DISCRETE_SEQ
                    )
                    fig.update_layout(xaxis_title="", yaxis_title="Count")
                    safe_plot(fig, key=f"{page_name}-overall-ai-bar")
            else:
                st.info("AI flag exists but no values (skipped).")
        else:
            st.info("AI flag column not available (skipped).")

        # AI_MaxScore distribution (if present)
        if ai_maxscore_col and has(df, ai_maxscore_col):
            fig = px.histogram(
                df, x=ai_maxscore_col,
                title="AI_MaxScore Distribution",
                color_discrete_sequence=[BRAND_PURPLE]
            )
            fig.update_layout(xaxis_title="AI_MaxScore", yaxis_title="Count")
            safe_plot(fig, key=f"{page_name}-overall-ai-maxscore")

# ==============================
# MAIN
# ==============================
def main():
    inject_css()
    st.title("ANALYTICS DASHBOARD")

    page = st.sidebar.radio(
        "Dashboard Pages",
        list(PAGES.keys()),
        index=0
    )

    render_page(page)

if __name__ == "__main__":
    main()
