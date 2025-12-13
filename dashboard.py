# dashboard_powerbi.py
# ------------------------------------------------------------
# PowerBI-like dashboard with 5 in-page tabs
# Auto-pulls data from 5 worksheets and AUTO-DETECTS:
#   - section headers (from *_Avg (0–3) and *_Qn# columns)
#   - question numbers available per section
# Works even when each worksheet has different sections / Q counts.
#
# Excludes: Date, Duration, AI_MaxScore
# ------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import gspread
from google.oauth2.service_account import Credentials


# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Scored Dashboard", layout="wide")


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
            --panel:#0f172a;
            --text:#e5e7eb;
            --muted:#9ca3af;
            --border:rgba(148,163,184,0.22);
        }
        [data-testid="stAppViewContainer"]{
            background: radial-gradient(circle at top left, #0b1220 0%, #0b1220 35%, #070b14 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"]{
            background:#070b14;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] *{ color: var(--text) !important; }
        .block-container{ max-width: 1500px; padding-top: 1rem; padding-bottom: 2.4rem; }

        .hdr{
            background: linear-gradient(135deg, rgba(242,106,33,0.20), rgba(250,204,21,0.06), rgba(17,24,39,0.0));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.35);
            margin-bottom: 12px;
        }
        .hdr h1{ margin:0; font-size: 26px; color: var(--text); }
        .hdr p{ margin:6px 0 0; color: var(--muted); }

        .card{
            background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(15,23,42,0.90));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 14px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.28);
            margin-bottom: 12px;
        }

        .kpis{
            display:grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 12px;
        }
        .kpi{
            background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(15,23,42,0.88));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 12px;
        }
        .kpi .label{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }
        .kpi .value{ color: var(--text); font-size: 20px; font-weight: 800; }
        .kpi .sub{ color: var(--muted); font-size: 12px; margin-top: 2px; }

        button[role="tab"]{
            border-radius: 999px !important;
            padding: 8px 14px !important;
            margin-right: 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


# ==============================
# Google Sheets loader
# ==============================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


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


@st.cache_data(ttl=120, show_spinner=False)
def load_ws_df(spreadsheet_key: str, worksheet_name: str) -> pd.DataFrame:
    gc = gs_client()
    sh = gc.open_by_key(spreadsheet_key)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    header = [str(h).strip() for h in values[0]]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)


# ==============================
# Data helpers
# ==============================
AVG_SUFFIX = "_Avg (0–3)"
QCOL_RX = re.compile(r"^(?P<header>.+)_Qn(?P<qn>\d+)$")
AVG_RX  = re.compile(r"^(?P<header>.+)_Avg\s*\(0[–-]3\)$")
RANK_RX = re.compile(r"^(?P<header>.+)_RANK$")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in out.columns]
    return out


def remove_unwanted(df: pd.DataFrame, hide_ai_cols: bool = True) -> pd.DataFrame:
    drop = set()
    for c in df.columns:
        cl = c.strip().lower()
        if cl.startswith("date"):
            drop.add(c)
        if "duration" in cl:
            drop.add(c)
        if cl == "ai_maxscore":
            drop.add(c)
        if hide_ai_cols and cl in ("ai-suspected", "ai_suspected"):
            drop.add(c)
    return df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")


def infer_structure(df: pd.DataFrame):
    """
    Infers:
      headers: list[str]
      header_to_qns: dict[str, sorted list[int]]
      has_avg/has_rank per header
    """
    headers = set()
    header_to_qns = {}

    for c in df.columns:
        c = str(c).strip()

        m = QCOL_RX.match(c)
        if m:
            h = m.group("header").strip()
            qn = int(m.group("qn"))
            headers.add(h)
            header_to_qns.setdefault(h, set()).add(qn)
            continue

        m = AVG_RX.match(c)
        if m:
            h = m.group("header").strip()
            headers.add(h)
            header_to_qns.setdefault(h, set())
            continue

        m = RANK_RX.match(c)
        if m:
            h = m.group("header").strip()
            headers.add(h)
            header_to_qns.setdefault(h, set())
            continue

    headers = sorted(headers, key=lambda x: x.lower())
    header_to_qns = {h: sorted(list(qs)) for h, qs in header_to_qns.items()}
    return headers, header_to_qns


def build_heatmap_long(df: pd.DataFrame, headers: list[str], header_to_qns: dict) -> pd.DataFrame:
    rows = []
    for h in headers:
        qns = header_to_qns.get(h, [])
        for qn in qns:
            col = f"{h}_Qn{qn}"
            if col in df.columns:
                v = to_num(df[col]).mean(skipna=True)
                rows.append({"Header": h, "Q": f"Q{qn}", "Mean": float(v) if pd.notna(v) else np.nan})
    return pd.DataFrame(rows)


def avg_by_header(df: pd.DataFrame, headers: list[str]) -> pd.DataFrame:
    rows = []
    for h in headers:
        c = f"{h}{AVG_SUFFIX}"
        if c in df.columns:
            rows.append({"Header": h, "Avg": float(to_num(df[c]).mean(skipna=True))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Avg", ascending=False)
    return out


def rank_dist(df: pd.DataFrame) -> pd.DataFrame:
    if "Overall Rank" not in df.columns:
        return pd.DataFrame()
    return (
        df["Overall Rank"].fillna("Unknown").astype(str)
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Overall Rank", "Overall Rank": "Count"})
    )


def top_bottom(df: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Overall Total (0–24)" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    staff = "Staff ID" if "Staff ID" in df.columns else None
    cols = ["Overall Total (0–24)"] + ([staff] if staff else [])
    dd = df[cols].copy()
    dd["Overall Total (0–24)"] = to_num(dd["Overall Total (0–24)"])
    dd = dd.dropna(subset=["Overall Total (0–24)"]).sort_values("Overall Total (0–24)", ascending=False)
    top = dd.head(n).reset_index(drop=True)
    bot = dd.tail(n).sort_values("Overall Total (0–24)", ascending=True).reset_index(drop=True)
    return top, bot


# ==============================
# One tab renderer
# ==============================
def render_tab(title: str, worksheet_name: str, hide_ai_cols: bool):
    ss_key = st.secrets.get("GSHEETS_SPREADSHEET_KEY", "")
    if not ss_key:
        st.error("GSHEETS_SPREADSHEET_KEY is missing in secrets.")
        return

    with st.spinner(f"Loading '{worksheet_name}' from Google Sheets..."):
        try:
            df_raw = load_ws_df(ss_key, worksheet_name)
        except Exception as e:
            st.error(f"Failed to load worksheet '{worksheet_name}': {type(e).__name__}: {e}")
            return

    if df_raw.empty:
        st.warning(f"Worksheet '{worksheet_name}' is empty or has no data.")
        return

    df = clean_cols(df_raw)
    df = remove_unwanted(df, hide_ai_cols=hide_ai_cols)

    # Infer structure for THIS worksheet (so pages can be different)
    headers, header_to_qns = infer_structure(df)

    # Filters (apply per worksheet)
    with st.sidebar:
        st.markdown("### Filters")
        if "Overall Rank" in df.columns:
            ranks = sorted(df["Overall Rank"].fillna("Unknown").astype(str).unique().tolist())
            pick = st.multiselect(f"{title}: Overall Rank", ranks, default=ranks)
            df = df[df["Overall Rank"].fillna("Unknown").astype(str).isin(pick)]

        if "Staff ID" in df.columns:
            s = st.text_input(f"{title}: Search Staff ID", value="")
            if s.strip():
                df = df[df["Staff ID"].astype(str).str.contains(s.strip(), case=False, na=False)]

        if headers:
            selected_headers = st.multiselect(
                f"{title}: Section headers",
                headers,
                default=headers,
            )
        else:
            selected_headers = []

    st.markdown(
        f"""
        <div class="hdr">
            <h1>{title}</h1>
            <p>Source: worksheet <b>{worksheet_name}</b>. Dashboard auto-detects the headers & Qn columns in this sheet.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    n = len(df)
    mean_total = float(to_num(df["Overall Total (0–24)"]).mean()) if "Overall Total (0–24)" in df.columns else np.nan
    med_total  = float(to_num(df["Overall Total (0–24)"]).median()) if "Overall Total (0–24)" in df.columns else np.nan

    st.markdown('<div class="kpis">', unsafe_allow_html=True)
    kpi_card("Responses", f"{n:,}", "After filters")
    kpi_card("Overall Mean (0–24)", f"{mean_total:.2f}" if np.isfinite(mean_total) else "—", "")
    kpi_card("Overall Median (0–24)", f"{med_total:.2f}" if np.isfinite(med_total) else "—", "")
    kpi_card("Detected headers", f"{len(headers):,}", "In this worksheet")
    st.markdown("</div>", unsafe_allow_html=True)

    if not headers:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("I can’t detect section headers in this worksheet yet")
        st.write("Expected patterns like:")
        st.code("Some Header_Qn1, Some Header_Qn2, ... and/or Some Header_Avg (0–3), Some Header_RANK")
        st.write("Columns found (sample):")
        st.write(list(df.columns)[:80])
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Apply header selection (limit visuals to chosen headers)
    headers_viz = [h for h in headers if h in selected_headers]

    # Heatmap + Average bars
    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Heatmap: Mean score by Header × Question")
        hm = build_heatmap_long(df, headers_viz, header_to_qns)
        if hm.empty:
            st.info("No per-question columns found in this sheet (no *_Qn# columns).")
        else:
            # sort headers in the heatmap by current avg (if possible)
            av = avg_by_header(df, headers_viz)
            if not av.empty:
                order = av["Header"].tolist()
            else:
                order = headers_viz

            chart = (
                alt.Chart(hm)
                .mark_rect()
                .encode(
                    x=alt.X("Q:N", title="Question"),
                    y=alt.Y("Header:N", title="Header", sort=order),
                    color=alt.Color("Mean:Q", title="Mean (0–3)", scale=alt.Scale(domain=[0, 3])),
                    tooltip=["Header", "Q", alt.Tooltip("Mean:Q", format=".2f")],
                )
                .properties(height=min(42 * len(order), 560))
            )
            st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Column chart: Average by Header")
        av = avg_by_header(df, headers_viz)
        if av.empty:
            st.info("No *_Avg (0–3) columns found in this sheet.")
        else:
            b = (
                alt.Chart(av)
                .mark_bar()
                .encode(
                    y=alt.Y("Header:N", sort="-x", title="Header"),
                    x=alt.X("Avg:Q", title="Mean (0–3)", scale=alt.Scale(domain=[0, 3])),
                    tooltip=["Header", alt.Tooltip("Avg:Q", format=".2f")],
                )
                .properties(height=min(42 * len(headers_viz), 560))
            )
            st.altair_chart(b, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Rank + Top/Bottom
    a, b = st.columns([0.6, 1.4])
    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Overall Rank distribution")
        rd = rank_dist(df)
        if rd.empty:
            st.info("No 'Overall Rank' column found in this sheet.")
        else:
            pie = (
                alt.Chart(rd)
                .mark_arc(innerRadius=40)
                .encode(
                    theta="Count:Q",
                    color="Overall Rank:N",
                    tooltip=["Overall Rank", "Count"],
                )
                .properties(height=320)
            )
            st.altair_chart(pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top & Bottom performers (Overall Total)")
        top, bot = top_bottom(df, n=10)
        if top.empty and bot.empty:
            st.info("No 'Overall Total (0–24)' column found in this sheet.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Top 10")
                st.dataframe(top, use_container_width=True, height=320)
            with c2:
                st.caption("Bottom 10")
                st.dataframe(bot, use_container_width=True, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    # Table view (headers only)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Table view (headers only)")
    keep = []
    if "Staff ID" in df.columns: keep.append("Staff ID")
    if "Overall Total (0–24)" in df.columns: keep.append("Overall Total (0–24)")
    if "Overall Rank" in df.columns: keep.append("Overall Rank")

    for h in headers_viz:
        for suf in [AVG_SUFFIX, "_RANK"]:
            c = f"{h}{suf}"
            if c in df.columns:
                keep.append(c)

    show = df[keep].copy() if keep else df.copy()
    for c in show.columns:
        if c.endswith(AVG_SUFFIX) or c == "Overall Total (0–24)":
            show[c] = to_num(show[c])
    st.dataframe(show, use_container_width=True, height=460)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================
# App main: 5 tabs (PowerBI-like)
# ==============================
def main():
    inject_css()

    sheet_advisory = st.secrets.get("GSHEETS_WORKSHEET_NAME", "Advisory")
    sheet_tl       = st.secrets.get("GSHEETS_WORKSHEET_NAME1", "Thought Leadership")
    sheet_gm       = st.secrets.get("GSHEETS_WORKSHEET_NAME2", "Growth Mindset")
    sheet_net      = st.secrets.get("GSHEETS_WORKSHEET_NAME3", "Networking")
    sheet_infl     = st.secrets.get("GSHEETS_WORKSHEET_NAME4", "Influencingrelationship")

    st.sidebar.markdown("## Dashboard Controls")
    hide_ai_cols = st.sidebar.checkbox("Hide AI columns", value=True)

    tabs = st.tabs([
        "Thought Leadership",
        "Growth Mindset",
        "Networking & Advocacy",
        "Advisory Skills",
        "Influencing Relationships",
    ])

    with tabs[0]:
        render_tab("Thought Leadership", sheet_tl, hide_ai_cols)

    with tabs[1]:
        render_tab("Growth Mindset", sheet_gm, hide_ai_cols)

    with tabs[2]:
        render_tab("Networking & Advocacy", sheet_net, hide_ai_cols)

    with tabs[3]:
        render_tab("Advisory Skills", sheet_advisory, hide_ai_cols)

    with tabs[4]:
        render_tab("Influencing Relationships", sheet_infl, hide_ai_cols)


if __name__ == "__main__":
    main()
