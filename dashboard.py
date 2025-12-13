# dashboard_app.py
# ------------------------------------------------------------
# PowerBI-style dashboard (5 pages) from SCORED Google Sheet: "Advisory"
# - Heatmaps + KPIs + leaderboards
# - No dates, no duration, no AI_MaxScore
# - Uses headers only (no question text)
# ------------------------------------------------------------

import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional: Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# Charts (Altair is usually bundled with Streamlit)
import altair as alt


# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Advisory Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# Styling (PowerBI-ish)
# ==============================
def inject_css():
    st.markdown(
        """
        <style>
        :root{
            --primary:#F26A21;
            --bg:#0b1220;
            --card:#111827;
            --card2:#0f172a;
            --text:#e5e7eb;
            --muted:#9ca3af;
            --border:rgba(148,163,184,0.25);
        }

        [data-testid="stAppViewContainer"]{
            background: radial-gradient(circle at top left, #0b1220 0%, #0b1220 30%, #0a0f1c 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"]{
            background: #070b14;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] *{
            color: var(--text) !important;
        }

        .block-container{
            max-width: 1400px;
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
        }

        .hdr{
            background: linear-gradient(135deg, rgba(242,106,33,0.22), rgba(250,204,21,0.06), rgba(17,24,39,0.0));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.35);
            margin-bottom: 14px;
        }
        .hdr h1{
            margin: 0;
            font-size: 26px;
            line-height: 1.1;
            color: var(--text);
        }
        .hdr p{
            margin: 6px 0 0;
            color: var(--muted);
        }

        .kpi{
            background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.88));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 14px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.28);
        }
        .kpi .label{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }
        .kpi .value{ font-size: 22px; font-weight: 700; color: var(--text); }
        .kpi .sub{ color: var(--muted); font-size: 12px; margin-top: 2px; }

        .card{
            background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.90));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 14px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.28);
            margin-bottom: 12px;
        }

        /* Streamlit chart background tweaks */
        .stDataFrame, .stTable { background: transparent; }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================
# Secrets / Google Sheets
# ==============================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME", "Advisory")
SPREADSHEET_KEY = st.secrets.get("GSHEETS_SPREADSHEET_KEY", "")


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
    required = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id", "token_uri"]
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
def load_sheet_as_df(spreadsheet_key: str, worksheet_name: str) -> pd.DataFrame:
    if not spreadsheet_key:
        raise ValueError("GSHEETS_SPREADSHEET_KEY not set.")
    gc = gs_client()
    sh = gc.open_by_key(spreadsheet_key)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    return df


# ==============================
# Utilities
# ==============================
def clean_colname(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def find_best_col(df_cols: List[str], wanted: str) -> str | None:
    """
    Best-effort: exact match first, then case-insensitive match, then startswith match.
    """
    if wanted in df_cols:
        return wanted
    low = {c.lower(): c for c in df_cols}
    if wanted.lower() in low:
        return low[wanted.lower()]
    # try "startswith"
    for c in df_cols:
        if c.lower().startswith(wanted.lower()):
            return c
    return None


def hide_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    User asked: no dates, no duration, no AI_MaxScore.
    Also hide AI_score_max/AI_MaxScore & any date-ish columns if present.
    """
    drop = set()

    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("date", "duration_min", "duration", "ai_maxscore", "ai_score_max"):
            drop.add(c)
        # sometimes date comes as "Date " etc.
        if cl.startswith("date"):
            drop.add(c)
        if "duration" in cl:
            drop.add(c)

    return df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")


def detect_headers_from_scored(df: pd.DataFrame) -> List[str]:
    """
    We treat "headers" as the attribute base names that have: "<Header>_Avg (0–3)" columns.
    """
    headers = []
    for c in df.columns:
        if c.endswith("_Avg (0–3)"):
            headers.append(c.replace("_Avg (0–3)", ""))
    # stable order
    return sorted(headers)


def get_section_columns(
    df: pd.DataFrame,
    header_names: List[str],
    qns: Tuple[int, ...] = (1, 2, 3, 4),
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      - score_cols: per-question score cols for heatmap (Header_Qn1..)
      - avg_cols: Header_Avg (0–3)
      - rank_cols: Header_RANK
    Only includes columns that exist.
    """
    df_cols = list(df.columns)

    score_cols = []
    avg_cols = []
    rank_cols = []

    for h in header_names:
        avg_c = find_best_col(df_cols, f"{h}_Avg (0–3)")
        rk_c = find_best_col(df_cols, f"{h}_RANK")
        if avg_c:
            avg_cols.append(avg_c)
        if rk_c:
            rank_cols.append(rk_c)

        for qn in qns:
            sc_c = find_best_col(df_cols, f"{h}_Qn{qn}")
            if sc_c:
                score_cols.append(sc_c)

    return score_cols, avg_cols, rank_cols


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def as_long_heatmap_df(df: pd.DataFrame, header_names: List[str], qns: Tuple[int, ...]) -> pd.DataFrame:
    """
    Build long df for heatmap:
      rows: Header
      cols: Qn
      value: mean score (0-3)
    """
    rows = []
    for h in header_names:
        for qn in qns:
            c = f"{h}_Qn{qn}"
            if c in df.columns:
                v = to_numeric_safe(df[c]).mean(skipna=True)
                rows.append({"Header": h, "Question": f"Q{qn}", "MeanScore": v})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["MeanScore"] = out["MeanScore"].fillna(0)
    return out


def bar_avg_df(df: pd.DataFrame, header_names: List[str]) -> pd.DataFrame:
    rows = []
    for h in header_names:
        c = f"{h}_Avg (0–3)"
        if c in df.columns:
            rows.append({"Header": h, "AvgScore": to_numeric_safe(df[c]).mean(skipna=True)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["AvgScore"] = out["AvgScore"].fillna(0)
    out = out.sort_values("AvgScore", ascending=False)
    return out


def rank_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "Overall Rank" not in df.columns:
        return pd.DataFrame()
    return (
        df["Overall Rank"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Overall Rank", "Overall Rank": "Count"})
    )


def top_bottom_table(df: pd.DataFrame, metric_col: str, staff_col: str, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metric_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    dd = df[[staff_col]].copy() if staff_col in df.columns else pd.DataFrame({"Staff ID": [""] * len(df)})
    if staff_col not in dd.columns:
        dd["Staff ID"] = ""

    dd[metric_col] = to_numeric_safe(df[metric_col])

    dd = dd.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False)
    top = dd.head(n).reset_index(drop=True)
    bottom = dd.tail(n).sort_values(metric_col, ascending=True).reset_index(drop=True)
    return top, bottom


# ==============================
# The 5 pages (headers only)
# ==============================
PAGES: Dict[str, Dict[str, object]] = {
    "Thought Leadership": {
        "headers": [
            "Locally Anchored Visioning",
            "Innovation and Insight",
            "Execution Planning",
            "Cross-Functional Collaboration",
            "Follow-Through Discipline",
            "Learning-Driven Adjustment",
            "Result-Oriented Decision-Making",
        ],
        "qns": (1, 2, 3, 4),
    },
    "Growth Mindset": {
        "headers": [
            "LEARNING AGILITY",
            "DIGITAL SAVVY",
            "INNOVATION",
            "CONTEXTUAL INTELLIGENCE SYSTEM THINKING",
        ],
        "qns": (1, 2, 3, 4),
    },
    "Networking and Advocacy": {
        "headers": [
            "Strategic Positioning & Donor Fluency",
            "Power-Aware Stakeholder Mapping",
            "Equitable Allyship & Local Fronting",
            "Coalition Governance & Convening",
            "Community-Centered Messaging",
            "Evidence-Led Learning (Local Knowledge)",
            "Influence Without Authority",
            "Risk Management & Adaptive Communication",
        ],
        "qns": (1, 2, 3, 4),
    },
    "Advisory Skills": {
        "headers": [
            "Strategic & analytical thinking",
            "Credibility & trustworthiness",
            "Effective communication & influence",
            "Client & stakeholder focus",
            "Fostering collaboration & partnership",
            "Ensuring relevance & impact",
            "Solution orientation & adaptability",
            "Capacity strengthening & empowerment support",
        ],
        "qns": (1, 2, 3, 4),
    },
    "Influencing Relationships": {
        "headers": [
            "Strategic Positioning & Political Acumen",
            "Stakeholder Mapping & Engagement",
            "Evidence-Informed Advocacy",
            "Communication, Framing & Messaging",
            "Risk Awareness & Mitigation",
            "Coalition Building & Collaborative Action",
            "Adaptive Tactics & Channel Selection",
            "Integrity & Values-Based Influencing",
        ],
        "qns": (1, 2, 3),  # your influencing tool uses Q1..Q3 in the code you shared
    },
}


# ==============================
# Load data
# ==============================
def load_data_panel() -> pd.DataFrame:
    st.sidebar.markdown("### Data source")

    mode = st.sidebar.radio(
        "Choose source",
        ["Google Sheets (recommended)", "Upload Excel (fallback)"],
        index=0,
    )

    if mode == "Google Sheets (recommended)":
        ws_name = st.sidebar.text_input("Worksheet name", value=DEFAULT_WS_NAME)
        st.sidebar.caption("Make sure the spreadsheet is shared with your service account email.")
        if not SPREADSHEET_KEY:
            st.sidebar.error("GSHEETS_SPREADSHEET_KEY is missing in secrets.")
            return pd.DataFrame()

        try:
            df = load_sheet_as_df(SPREADSHEET_KEY, ws_name)
        except Exception as e:
            st.sidebar.error(f"Failed to read Google Sheet: {type(e).__name__}: {e}")
            return pd.DataFrame()

        return df

    # Fallback: upload Excel
    up = st.sidebar.file_uploader("Upload Advisory.xlsx", type=["xlsx"])
    if not up:
        st.sidebar.info("Upload the scored Excel if Sheets is unavailable.")
        return pd.DataFrame()
    try:
        df = pd.read_excel(up)
        return df
    except Exception as e:
        st.sidebar.error(f"Failed to read Excel: {type(e).__name__}: {e}")
        return pd.DataFrame()


# ==============================
# Main rendering
# ==============================
def render_page(df_raw: pd.DataFrame, page_name: str):
    page_spec = PAGES[page_name]
    wanted_headers: List[str] = page_spec["headers"]  # type: ignore
    qns: Tuple[int, ...] = page_spec["qns"]  # type: ignore

    # Clean colnames
    df = df_raw.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    # Remove unwanted columns (Date, Duration, AI_MaxScore)
    df = hide_unwanted_columns(df)

    # Try to hide AI columns by default too (not requested explicitly, but usually desired)
    hide_ai = st.sidebar.checkbox("Hide AI columns", value=True)
    if hide_ai:
        df = df.drop(columns=[c for c in df.columns if c.lower() in ("ai_score_max", "ai_maxscore", "ai_suspected", "ai-suspected")], errors="ignore")

    # Filters
    st.sidebar.markdown("### Filters")
    staff_col = "Staff ID" if "Staff ID" in df.columns else None
    overall_total_col = "Overall Total (0–24)" if "Overall Total (0–24)" in df.columns else None

    # Overall Rank filter
    if "Overall Rank" in df.columns:
        ranks = sorted(df["Overall Rank"].fillna("Unknown").astype(str).unique().tolist())
        sel_ranks = st.sidebar.multiselect("Overall Rank", ranks, default=ranks)
        df = df[df["Overall Rank"].fillna("Unknown").astype(str).isin(sel_ranks)]

    # Staff ID search
    if staff_col:
        q = st.sidebar.text_input("Search Staff ID", value="")
        if q.strip():
            df = df[df[staff_col].astype(str).str.contains(q.strip(), case=False, na=False)]

    # Header block
    inject_css()
    st.markdown(
        f"""
        <div class="hdr">
            <h1>{page_name}</h1>
            <p>Scored data from Google Sheets worksheet: <b>{DEFAULT_WS_NAME}</b> • Heatmaps • Averages • Rankings</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    n = len(df)
    overall_mean = float(to_numeric_safe(df[overall_total_col]).mean()) if overall_total_col else np.nan
    overall_med = float(to_numeric_safe(df[overall_total_col]).median()) if overall_total_col else np.nan

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Responses", f"{n:,}", "Rows after filters")
    with k2:
        kpi_card("Overall mean (0–24)", f"{overall_mean:.2f}" if np.isfinite(overall_mean) else "—", "")
    with k3:
        kpi_card("Overall median (0–24)", f"{overall_med:.2f}" if np.isfinite(overall_med) else "—", "")
    with k4:
        # count “Exemplary…” if exists
        ex = 0
        if "Overall Rank" in df.columns:
            ex = int((df["Overall Rank"].astype(str) == "Exemplary Thought Leader").sum())
        kpi_card("Exemplary count", f"{ex:,}", "If rank exists")

    # Determine which of the requested headers actually exist in this dataset
    available_headers = []
    for h in wanted_headers:
        if f"{h}_Avg (0–3)" in df.columns or any(f"{h}_Qn{qn}" in df.columns for qn in qns):
            available_headers.append(h)

    # If nothing matches, show hint + show detected headers
    if not available_headers:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("No matching headers found in this sheet")
        detected = detect_headers_from_scored(df)
        st.write("Detected headers (from *_Avg (0–3) columns):")
        st.write(detected[:60])
        st.info("This dashboard expects your scored sheet columns to be like: 'Header_Qn1', 'Header_Avg (0–3)', 'Header_RANK'.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Charts row: Heatmap + Bar chart
    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Heatmap: mean score by header and question")
        hm = as_long_heatmap_df(df, available_headers, qns)
        if hm.empty:
            st.info("No per-question columns found for these headers.")
        else:
            heat = (
                alt.Chart(hm)
                .mark_rect()
                .encode(
                    x=alt.X("Question:N", title="Question"),
                    y=alt.Y("Header:N", sort=available_headers, title="Header"),
                    color=alt.Color("MeanScore:Q", title="Mean (0–3)", scale=alt.Scale(domain=[0, 3])),
                    tooltip=["Header", "Question", alt.Tooltip("MeanScore:Q", format=".2f")],
                )
                .properties(height=min(40 * len(available_headers), 520))
            )
            st.altair_chart(heat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Average score by header")
        bd = bar_avg_df(df, available_headers)
        if bd.empty:
            st.info("No *_Avg (0–3) columns found for these headers.")
        else:
            bar = (
                alt.Chart(bd)
                .mark_bar()
                .encode(
                    y=alt.Y("Header:N", sort="-x", title="Header"),
                    x=alt.X("AvgScore:Q", title="Mean (0–3)", scale=alt.Scale(domain=[0, 3])),
                    tooltip=["Header", alt.Tooltip("AvgScore:Q", format=".2f")],
                )
                .properties(height=min(40 * len(available_headers), 520))
            )
            st.altair_chart(bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Distributions + Leaderboard
    c3, c4 = st.columns([0.65, 1.35])

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Overall rank distribution")
        rd = rank_distribution(df)
        if rd.empty:
            st.info("No 'Overall Rank' column found.")
        else:
            pie = (
                alt.Chart(rd)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color("Overall Rank:N"),
                    tooltip=["Overall Rank", "Count"],
                )
                .properties(height=320)
            )
            st.altair_chart(pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top / bottom performers (Overall Total)")
        if not overall_total_col:
            st.info("No 'Overall Total (0–24)' column found.")
        else:
            top, bottom = top_bottom_table(df, overall_total_col, staff_col or "Staff ID", n=10)
            t1, t2 = st.columns(2)
            with t1:
                st.caption("Top 10")
                st.dataframe(top, use_container_width=True, height=320)
            with t2:
                st.caption("Bottom 10")
                st.dataframe(bottom, use_container_width=True, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    # Data table (selected columns only)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Table view (selected columns only)")

    # Build a clean table selection: Staff ID, Overall, then header averages + ranks
    keep = []
    if "Staff ID" in df.columns:
        keep.append("Staff ID")
    if "Overall Total (0–24)" in df.columns:
        keep.append("Overall Total (0–24)")
    if "Overall Rank" in df.columns:
        keep.append("Overall Rank")

    for h in available_headers:
        for suf in ["_Avg (0–3)", "_RANK"]:
            c = f"{h}{suf}"
            if c in df.columns:
                keep.append(c)

    show_df = df[keep].copy() if keep else df.copy()

    # numeric cleanup
    for c in show_df.columns:
        if c.endswith("_Avg (0–3)") or c == "Overall Total (0–24)":
            show_df[c] = to_numeric_safe(show_df[c])

    st.dataframe(show_df, use_container_width=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    df_raw = load_data_panel()

    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Dashboard page", list(PAGES.keys()), index=0)

    if df_raw is None or df_raw.empty:
        inject_css()
        st.markdown(
            """
            <div class="hdr">
              <h1>Advisory Dashboard</h1>
              <p>No data loaded yet. Choose Google Sheets or upload your scored Excel.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    render_page(df_raw, page)


if __name__ == "__main__":
    main()
