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
# Rules respected:
# - Pull from Google Sheets using gcp_service_account + GSHEETS_SPREADSHEET_KEY
# - DO NOT show questions (only short titles you define)
# - NO Date / Duration fields
# - Uses AI_Suspected (and shows AI analysis in Overall tab)
# - Heatmaps + donut/pie + bar/column + histogram
# - If data/columns missing: skip silently (no crashing)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Scoring Dashboard", layout="wide")

# Your worksheet names (exact)
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
# SECTION DEFINITIONS (edit titles anytime)
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
        "ai_maxscore_col": "AI_MaxScore",  # optional if present
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
        "sections": [
            "Learning Agility",
            "Digital Savvy",
            "Innovation",
            "Contextual Intelligence",
        ],
        "overall_total_col": "Overall Total (0–12)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI_Suspected",
        "ai_maxscore_col": "AI_MaxScore",  # optional
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
        "ai_col": "AI_Suspected",          # sometimes sheet has AI-Suspected
        "ai_maxscore_col": "AI_MaxScore",  # often present here
        "question_titles": {},  # optional
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
        "ai_maxscore_col": "AI_MaxScore",  # optional
        "question_titles": {},  # optional
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
        "ai_col": "AI_Suspected",          # sometimes sheet has AI-Suspected
        "ai_maxscore_col": "AI_MaxScore",  # often present here
        "question_titles": {},  # optional
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
# CLEANING / SAFETY
# ==============================
def drop_date_duration_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column that smells like duration/date/time; keep everything else."""
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

# ==============================
# VIS HELPERS
# ==============================
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
# RENDER ONE PAGE (worksheet)
# ==============================
def render_page(page_name: str):
    cfg = PAGES[page_name]
    ws_name = cfg["worksheet"]

    with st.spinner(f"Loading: {ws_name} ..."):
        df = load_sheet_df(ws_name)

    # Drop Date/Duration columns (rule)
    df = drop_date_duration_cols(df)

    st.caption(f"Worksheet: {ws_name} • Rows: {len(df):,} • Columns: {len(df.columns):,}")

    # Build tabs: one per section + Overall
    section_tabs = cfg["sections"] + ["Overall"]
    tabs = st.tabs(section_tabs)

    # Per-section tabs
    for i, section in enumerate(cfg["sections"]):
        with tabs[i]:
            st.subheader(section)

            avg_col = f"{section}_Avg (0–3)"
            rnk_col = f"{section}_RANK"

            c1, c2 = st.columns(2)
            with c1:
                if has(df, avg_col):
                    fig = px.histogram(df, x=avg_col, nbins=10, title="Section Average Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Section average not available (skipped).")

            with c2:
                if has(df, rnk_col):
                    r = _clean_series(df[rnk_col])
                    if len(r):
                        r_df = r.value_counts().reset_index()
                        r_df.columns = ["Rank", "Count"]
                        fig = px.pie(r_df, names="Rank", values="Count", hole=0.55, title="Section Rank (Donut)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Section rank has no values (skipped).")
                else:
                    st.info("Section rank not available (skipped).")

            st.divider()

            # NOTE: We DO NOT show the original questions (rule).
            # We only show your short titles.
            q_titles = cfg.get("question_titles", {}).get(section, [])

            for qn in range(1, 5):
                score_col  = f"{section}_Qn{qn}"
                rubric_col = f"{section}_Rubric_Qn{qn}"

                # If both missing -> skip quietly
                if (not has(df, score_col)) and (not has(df, rubric_col)):
                    continue

                title = q_titles[qn - 1] if (q_titles and len(q_titles) >= qn) else f"Qn{qn}"
                st.markdown(f"### {title}")

                a, b = st.columns(2)

                with a:
                    if has(df, score_col):
                        d = score_dist(df, score_col)
                        fig = px.bar(d, x="Score", y="Percent", text="Percent", title="Score Distribution (%)")
                        fig.update_traces(texttemplate="%{text}%", textposition="outside")
                        fig.update_layout(yaxis_title="Percent", xaxis_title="Score (0–3)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Score column missing (skipped).")

                with b:
                    if has(df, rubric_col):
                        rf = rubric_freq(df, rubric_col)
                        if len(rf):
                            fig = px.bar(rf, x="Count", y="Rubric", orientation="h", title="Rubric Frequency")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Rubric has no values (skipped).")
                    else:
                        st.info("Rubric column missing (skipped).")

                st.divider()

            st.markdown("### Heatmap (Score % by Question)")
            try:
                h = section_heatmap(df, section, q_titles)
                if not h.empty:
                    fig = px.imshow(h, labels=dict(x="Score", y="Question", color="%"), title=f"{section} Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Heatmap skipped (missing Qn columns or no values).")
            except Exception:
                # rule: no scary errors; just skip
                st.info("Heatmap skipped.")

    # Overall tab
    with tabs[-1]:
        st.subheader("Overall")

        overall_total = cfg["overall_total_col"]
        overall_rank  = cfg["overall_rank_col"]

        # AI columns can vary across your sheets:
        ai_col = first_existing(df, [cfg.get("ai_col"), "AI_Suspected", "AI-Suspected"])
        ai_maxscore_col = first_existing(df, [cfg.get("ai_maxscore_col"), "AI_MaxScore"])

        c1, c2 = st.columns(2)

        with c1:
            if has(df, overall_total):
                fig = px.histogram(df, x=overall_total, title="Overall Total Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Overall Total not available (skipped).")

        with c2:
            if has(df, overall_rank):
                r = _clean_series(df[overall_rank])
                if len(r):
                    r_df = r.value_counts().reset_index()
                    r_df.columns = ["Rank", "Count"]
                    fig = px.pie(r_df, names="Rank", values="Count", hole=0.55, title="Overall Rank (Donut)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Overall Rank has no values (skipped).")
            else:
                st.info("Overall Rank not available (skipped).")

        st.divider()

        # AI analysis (Donut + Bar)
        if ai_col and has(df, ai_col):
            ai = _clean_series(df[ai_col])
            if len(ai):
                ai_df = ai.value_counts().reset_index()
                ai_df.columns = ["AI", "Count"]

                c3, c4 = st.columns(2)
                with c3:
                    fig = px.pie(ai_df, names="AI", values="Count", hole=0.55, title="AI Flag (Donut)")
                    st.plotly_chart(fig, use_container_width=True)
                with c4:
                    fig = px.bar(ai_df, x="AI", y="Count", title="AI Flag (Counts)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("AI column exists but no values (skipped).")
        else:
            st.info("AI flag column not available (skipped).")

        # AI_MaxScore (if present) — histogram
        if ai_maxscore_col and has(df, ai_maxscore_col):
            st.divider()
            fig = px.histogram(df, x=ai_maxscore_col, title="AI_MaxScore Distribution")
            st.plotly_chart(fig, use_container_width=True)

# ==============================
# MAIN
# ==============================
def main():
    st.title("PowerBI-style Scoring Dashboard")

    page = st.sidebar.radio(
        "Dashboard Pages",
        list(PAGES.keys()),
        index=0
    )

    render_page(page)

if __name__ == "__main__":
    main()
