# file: thoughtleadership.py — Working Kobo mapping + NEW exemplar-nearest scoring

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

import json, re, unicodedata
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process


# ==============================
# UI / STYLING
# ==============================
def inject_css():
    st.markdown("""
        <style>
        :root {
            /* Brand colours */
            --primary: #F26A21;            /* CARE orange */
            --primary-soft: #FDE7D6;
            --primary-soft-stronger: #FBD0AD;

            --gold: #FACC15;
            --gold-soft: #FEF9C3;
            --silver: #E5E7EB;

            --bg-main: #f5f5f5;
            --card-bg: #ffffff;
            --text-main: #111827;
            --text-muted: #6b7280;
            --border-subtle: #e5e7eb;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #FFF7ED 0, #F9FAFB 40%, #F3F4F6 100%);
            color: var(--text-main);
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #1f2937;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] * { color: #e5e7eb !important; }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1, h2, h3 {
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-main);
        }
        h1 { font-size: 2.1rem; font-weight: 700; }
        h2 { margin-top: 1.5rem; font-size: 1.3rem; }
        p, span, label { color: var(--text-muted); }

        .app-header-card {
            position: relative;
            background:
                radial-gradient(circle at top left,
                    rgba(242,106,33,0.15),
                    rgba(250,204,21,0.06),
                    #ffffff);
            border-radius: 1.25rem;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.12);
            margin-bottom: 1.4rem;
            overflow: hidden;
        }
        .app-header-card::before {
            content: "";
            position: absolute;
            inset: 0;
            height: 3px;
            background: linear-gradient(90deg,
                var(--gold-soft),
                var(--primary),
                var(--silver),
                var(--gold));
            opacity: 0.95;
        }
        .app-header-card::after {
            content: "";
            position: absolute;
            bottom: -40px;
            right: -40px;
            width: 140px;
            height: 140px;
            background: radial-gradient(circle,
                rgba(250,204,21,0.35),
                transparent 60%);
            opacity: 0.7;
        }
        .app-header-subtitle { font-size: 0.9rem; color: var(--text-muted); }

        .pill {
            display: inline-block;
            font-size: 0.75rem;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            background: rgba(242,106,33,0.08);
            border: 1px solid rgba(242,106,33,0.6);
            color: #9A3412;
            margin-bottom: 0.4rem;
        }

        .section-card {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border-subtle);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.04);
        }

        .stDataFrame table {
            font-size: 13px;
            border-radius: 0.75rem;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
        }
        .stDataFrame table thead tr th {
            background-color: var(--primary-soft);
            font-weight: 600;
            color: #7c2d12;
        }

        .stDownloadButton button, .stButton button {
            border-radius: 999px !important;
            padding: 0.35rem 1.2rem !important;
            font-weight: 600 !important;
            border: 1px solid rgba(242,106,33,0.85) !important;
            background: linear-gradient(135deg, var(--primary) 0%, #FB923C 100%) !important;
            color: #FFFBEB !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
            box-shadow: 0 12px 25px rgba(248,113,22,0.45);
        }

        .stAlert { border-radius: 0.8rem; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)


# ==============================
# SECRETS / PATHS
# ==============================
KOBO_BASE       = st.secrets.get("KOBO_BASE", "https://kobo.care.org")
KOBO_ASSET_ID1  = st.secrets.get("KOBO_ASSET_ID1", "")
KOBO_TOKEN      = st.secrets.get("KOBO_TOKEN", "")
AUTO_PUSH       = bool(st.secrets.get("AUTO_PUSH", False))

DATASETS_DIR    = Path("DATASETS")
MAPPING_PATH    = DATASETS_DIR / "mapping1.csv"
EXEMPLARS_PATH  = DATASETS_DIR / "thought_leadership.cleaned.jsonl"


# ==============================
# CONSTANTS
# ==============================
BANDS = {0:"Counterproductive",1:"Compliant",2:"Strategic",3:"Transformative"}
OVERALL_BANDS = [
    ("Exemplary Thought Leader", 19, 21),
    ("Strategic Advisor",       14, 18),
    ("Emerging Advisor",         8, 13),
    ("Needs Capacity Support",   0,  7),
]

ORDERED_ATTRS = [
    "Locally Anchored Visioning",
    "Innovation and Insight",
    "Execution Planning",
    "Cross-Functional Collaboration",
    "Follow-Through Discipline",
    "Learning-Driven Adjustment",
    "Result-Oriented Decision-Making",
]

FUZZY_THRESHOLD = 80
MIN_QA_OVERLAP  = 0.05

# reuse score if answer is semantically same for a question
DUP_SIM = float(st.secrets.get("DUP_SIM", 0.92))

# exemplar-nearest scoring controls
KNN_METHOD   = st.secrets.get("KNN_METHOD", "top1").lower().strip()  # "top1" or "softmax"
KNN_K        = int(st.secrets.get("KNN_K", 7))
KNN_TEMP     = float(st.secrets.get("KNN_TEMP", 0.08))
CONF_CLAMP   = float(st.secrets.get("CONF_CLAMP", 0.0))  # 0 disables; e.g. 0.45 clamps high scores on low confidence

# rubric override controls (post-score validator)
RUBRIC_OVERRIDE = bool(st.secrets.get("RUBRIC_OVERRIDE", True))
RUBRIC_UPGRADE_CONF = float(st.secrets.get("RUBRIC_UPGRADE_CONF", 0.75))
RUBRIC_DOWNGRADE = bool(st.secrets.get("RUBRIC_DOWNGRADE", False))
RUBRIC_SHOW_AUDIT = bool(st.secrets.get("RUBRIC_SHOW_AUDIT", False))

PASSTHROUGH_HINTS = [
    "staff id","staff_id","staffid","_id","id","_uuid","uuid","instanceid","_submission_time",
    "submissiondate","submission_date","start","_start","end","_end","today","date","deviceid",
    "username","enumerator","submitted_via_web","_xform_id_string","formid","assetid"
]

_EXCLUDE_SOURCE_COLS_LOWER = {
    "_id","formhub/uuid","start","end","today","staff_id","meta/instanceid",
    "_xform_id_string","_uuid","meta/rootuuid","_submission_time","_validation_status"
}


# ==============================
# AI DETECTION (keep your working logic)
# ==============================
AI_SUSPECT_THRESHOLD = float(st.secrets.get("AI_SUSPECT_THRESHOLD", 0.60))

TRANSITION_OPEN_RX = re.compile(
    r"^(?:first|second|third|finally|moreover|additionally|furthermore|however|therefore|in conclusion)\b",
    re.I
)
LIST_CUES_RX       = re.compile(r"\b(?:first|second|third|finally)\b", re.I)
BULLET_RX          = re.compile(r"^[-*•]\s", re.M)
LONG_DASH_HARD_RX  = re.compile(r"[—–]")
SYMBOL_RX = re.compile(
    r"[—–\-_]{2,}"
    r"|[≥≤≧≦≈±×÷%]"
    r"|[→←⇒↔↑↓]"
    r"|[•●◆▶✓✔✗❌§†‡]",
    re.U
)
TIMEBOX_RX = re.compile(
    r"(?:\bday\s*\d+\b|\bweek\s*\d+\b|\bmonth\s*\d+\b|\bquarter\s*\d+\b"
    r"|\b\d+\s*(?:days?|weeks?|months?|quarters?)\b|\bby\s+day\s*\d+\b)",
    re.I
)
AI_RX = re.compile(r"(?:as an ai\b|i am an ai\b)", re.I)
DAY_RANGE_RX        = re.compile(r"\bday\s*\d+\s*[-–]\s*\d+\b", re.I)
PIPE_LIST_RX        = re.compile(r"\s\|\s")
PARENS_ACRONYMS_RX  = re.compile(r"\(([A-Z]{2,}(?:s)?(?:\s*,\s*[A-Z]{2,}(?:s)?)+).*?\)")
NUMBERED_BULLETS_RX = re.compile(r"\b\d+\s*[\.\)]\s*")
SLASH_PAIR_RX       = re.compile(r"\b\w+/\w+\b")

AI_BUZZWORDS = {
    "minimum viable", "feedback loop", "trade-off", "evidence-based",
    "stakeholder alignment", "learners' agency", "learners’ agency",
    "norm shifts", "quick win", "low-lift", "scalable",
    "best practice", "pilot theatre", "timeboxed"
}


# ==============================
# HELPERS
# ==============================
def clean(s):
    if s is None:
        return ""
    try:
        if isinstance(s, float) and s != s:
            return ""
    except Exception:
        pass
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()

def qa_overlap(ans, qtext) -> float:
    def _t(x) -> str:
        if x is None:
            return ""
        try:
            if isinstance(x, float) and x != x:
                return ""
        except Exception:
            pass
        return str(x)

    ans_s = _t(ans).lower()
    q_s   = _t(qtext).lower()

    at = set(re.findall(r"\w+", ans_s))
    qt = set(re.findall(r"\w+", q_s))
    return (len(at & qt) / (len(qt) + 1.0)) if qt else 1.0

def ai_signal_score(text: str, question_hint: str = "") -> float:
    t = clean(text)
    if not t:
        return 0.0
    if LONG_DASH_HARD_RX.search(t):
        return 1.0

    score = 0.0
    if SYMBOL_RX.search(t):               score += 0.35
    if TIMEBOX_RX.search(t):              score += 0.15
    if AI_RX.search(t):                   score += 0.35
    if TRANSITION_OPEN_RX.search(t):      score += 0.12
    if LIST_CUES_RX.search(t):            score += 0.12
    if BULLET_RX.search(t):               score += 0.08

    if DAY_RANGE_RX.search(t):            score += 0.15
    if PIPE_LIST_RX.search(t):            score += 0.10
    if PARENS_ACRONYMS_RX.search(t):      score += 0.10
    if NUMBERED_BULLETS_RX.search(t):     score += 0.12
    if SLASH_PAIR_RX.search(t):           score += 0.08

    hits = 0
    for rx in (TIMEBOX_RX, DAY_RANGE_RX, PIPE_LIST_RX, NUMBERED_BULLETS_RX):
        if rx.search(t):
            hits += 1
    if hits >= 2: score += 0.25
    if hits >= 3: score += 0.15

    tl = t.lower()
    buzz_hits = sum(1 for b in AI_BUZZWORDS if b in tl)
    if buzz_hits:
        score += min(0.24, 0.08 * buzz_hits)

    if question_hint:
        overlap = qa_overlap(t, question_hint)
        if overlap < 0.06:
            score += 0.10

    return max(0.0, min(1.0, score))


# ==============================
# RUBRIC VALIDATOR (post-score)
# ==============================
# This layer checks whether an answer contains the *evidence* expected by the 0–3 anchors
# for each question, and can upgrade (or optionally downgrade) the exemplar score.

_WORD_RX = re.compile(r"\w+")
def _has_any(text: str, patterns: list[str]) -> bool:
    t = (text or "").lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False

def _count_any(text: str, patterns: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for p in patterns if re.search(p, t))

def rubric_validate(qid: str, ans: str) -> tuple[int, float, str]:
    """Return (rubric_score 0..3, rubric_conf 0..1, reason)."""
    a = clean(ans)
    if not a:
        return 0, 0.90, "empty"

    # Counterproductive / derailing cues (keep conservative)
    harmful = _has_any(a, [
        r"\bignore (the )?community\b",
        r"\bdrop (the )?gender\b",
        r"\bremove women\b",
        r"\bonly (work with )?big ngos?\b",
        r"\bjust scale fast\b",
        r"\bcentraliz(e|ing)\b.*\bcommunity\b",
    ])
    if harmful:
        return 0, 0.90, "harmful/derailing"

    # Very short answers: rarely deserve 2/3
    if len(a) < 18 or len(_WORD_RX.findall(a)) < 4:
        return 1, 0.75, "too_short"

    qid = (qid or "").strip().upper()

    # ---- Shared “execution evidence” signals (used across many questions) ----
    has_time     = _has_any(a, [r"\b60 days\b", r"\b90 days\b", r"\bby friday\b", r"\bweekly\b", r"\bbi\-?weekly\b", r"\bquarterly\b", r"\b\d+\s*(days?|weeks?|months?)\b"])
    has_owner    = _has_any(a, [r"\bowner\b", r"\baccountable\b", r"\bresponsible\b", r"\blead\b", r"\bby name\b", r"\bRACI\b", r"\bescalat"])
    has_artifact = _has_any(a, [r"\bdashboard\b", r"\bdecision memo\b", r"\blearning log\b", r"\binsight brief\b", r"\bscorecard\b", r"\bbacklog\b", r"\bagenda\b", r"\boutputs\b", r"\bchecklist\b", r"\bplaybook\b"])
    has_gate     = _has_any(a, [r"\bdecision gate\b", r"\bgate\b", r"\bthreshold\b", r"\btrigger\b", r"\bif .* then\b"])
    has_tradeoff = _has_any(a, [r"\btrade\-?off\b", r"\bnon\-?negotiable\b", r"\bflex\b", r"\bguardrail\b", r"\bsafeguard\b", r"\bprotect\b"])
    has_local    = _has_any(a, [r"\bgrassroots\b", r"\bcommunity voice\b", r"\blocal leadership\b", r"\bkujenga\b", r"\bparish\b", r"\bvsla(s)?\b", r"\bproducer groups?\b"])
    has_gender   = _has_any(a, [r"\bwomen\b", r"\bgender\b", r"\bequity\b", r"\binclusion\b"])

    # helpers for “transformative safeguards”
    safeguards = _has_any(a, [
        r"\btor(s)?\b|\bterms of reference\b",
        r"\bbudget line\b|\bring\-?fence\b|\bprotected funding\b|\bpercentage\b.*\bbudget\b",
        r"\bgovernance\b|\bsteering committee\b|\bdecision rights?\b|\bco\-?own(ed)?\b",
        r"\bscorecard\b|\bgrievance\b|\baccountability\b|\bsigned changes\b",
    ])

    # =====================
    # LAV (A1)
    # =====================
    if qid == "LAV_Q1":
        core = _has_any(a, [r"women\-?led", r"producer groups?", r"local buying", r"buying days?", r"co\-?design", r"vsla(s)?", r"grassroots", r"community"])
        flex = _has_any(a, [r"\bflex\b", r"which districts?", r"sequence", r"phased", r"onboard", r"aggregation", r"parish hubs?", r"mobile collection"])
        if core and flex and safeguards:
            return 3, 0.80, "core+flex+safeguards"
        if core and flex:
            return 2, 0.78, "core+flex"
        if core or flex:
            return 1, 0.65, "partial(core/flex)"
        return 1, 0.55, "generic"

    if qid == "LAV_Q2":
        # Needs ToR + budget + local leadership
        tor = _has_any(a, [r"\btor(s)?\b", r"terms of reference"])
        bud = _has_any(a, [r"\bbudget\b", r"\bbudget line\b", r"ring\-?fence", r"protected funding"])
        local = _has_any(a, [r"local leadership", r"community", r"women\-?led", r"kujenga", r"vsla"])
        enforce = _has_any(a, [r"decision rights?", r"sign\-?off", r"accountability", r"scorecard", r"minimum percentage"])
        if tor and bud and local and enforce:
            return 3, 0.80, "tor+budget+local+enforcement"
        if tor and bud and local:
            return 2, 0.75, "tor+budget+local"
        if local and (tor or bud):
            return 1, 0.60, "mentions local + (tor or budget)"
        return 1, 0.55, "generic"

    if qid == "LAV_Q3":
        # Needs risks of formal partners + safeguards to protect voice
        risks = _has_any(a, [r"risk", r"sidelin(e|ed)", r"capture", r"elite", r"dilut", r"token"])
        formal = _has_any(a, [r"formal partners?", r"big ngos?", r"large ngo", r"prime partner"])
        voice = _has_any(a, [r"community voice", r"local leadership", r"women", r"grassroots"])
        if (formal or risks) and voice and safeguards:
            return 3, 0.78, "risks+voice+safeguards"
        if (formal or risks) and voice:
            return 2, 0.72, "risks+voice"
        if formal or risks:
            return 1, 0.60, "mentions risk/formal only"
        return 1, 0.55, "generic"

    if qid == "LAV_Q4":
        # Trade ambition for protection, with reasoning
        trade = _has_any(a, [r"trade\-?off", r"i would trade", r"drop", r"reduce", r"slow", r"phase"])
        protect = _has_any(a, [r"protect", r"non\-?negotiable", r"safeguard", r"guardrail"])
        why = _has_any(a, [r"because", r"so that", r"in order to"])
        if trade and protect and why and (has_local or has_gender):
            return 3, 0.75, "explicit trade-off + protection + reasoning"
        if trade and protect and why:
            return 2, 0.70, "trade-off + reasoning"
        if trade or protect:
            return 1, 0.60, "partial trade/protect"
        return 1, 0.55, "generic"

    # =====================
    # II (A2)
    # =====================
    if qid == "II_Q1":
        # field-first learning loop: sources, cadence, artifacts, trigger
        sources = _has_any(a, [r"women", r"vsla", r"farmers", r"traders?", r"district", r"extension", r"kujenga", r"community"])
        cadence = has_time or _has_any(a, [r"cadence", r"weekly", r"bi\-?weekly", r"daily"])
        artifacts = has_artifact or _has_any(a, [r"learning loop", r"log", r"brief", r"dashboard", r"report"])
        trigger = has_gate or _has_any(a, [r"trigger", r"if", r"when .* then", r"design sprint", r"sprint"])
        if sources and cadence and artifacts and trigger and safeguards:
            return 3, 0.78, "loop+trigger+artifacts(+safeguards)"
        if sources and cadence and artifacts and trigger:
            return 2, 0.75, "loop+trigger+artifacts"
        if sources and (cadence or artifacts):
            return 1, 0.60, "partial learning loop"
        return 1, 0.55, "generic"

    if qid == "II_Q2":
        insight = _has_any(a, [r"insight", r"we expect", r"contradict", r"assumption", r"hq"])
        women_src = has_gender or _has_any(a, [r"women's groups?", r"women\-?led", r"vsla"])
        act = _has_any(a, [r"act", r"decide", r"change", r"adjust", r"within", r"fast", r"quick"])
        mechanism = has_artifact or has_gate or _has_any(a, [r"feedback", r"listening", r"session", r"review", r"stand\-?up"])
        if insight and women_src and act and mechanism and (has_owner or has_time):
            return 3, 0.75, "concrete insight + fast action loop"
        if insight and women_src and (act or mechanism):
            return 2, 0.68, "insight + action/mechanism"
        if women_src and (insight or mechanism):
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "II_Q3":
        policy = _has_any(a, [r"policy", r"ministry", r"agro\-?parks?", r"agro industrial", r"government"])
        experiment = _has_any(a, [r"experiment", r"test", r"pilot", r"iterate", r"sprint"])
        anti_theatre = _has_any(a, [r"pilot theatre", r"theatre", r"real decisions", r"decision gate", r"scale criteria", r"threshold"])
        if policy and experiment and anti_theatre and (has_gate or has_owner):
            return 3, 0.75, "policy+experimentation+decision gates"
        if (policy or experiment) and anti_theatre:
            return 2, 0.68, "anti-theatre + tension addressed"
        if policy or experiment:
            return 1, 0.60, "mentions one side"
        return 1, 0.55, "generic"

    if qid == "II_Q4":
        frugal = _has_any(a, [r"frugal", r"low\-?cost", r"cheap", r"simple", r"one week", r"within a week", r"test"])
        market = _has_any(a, [r"market access", r"buyers?", r"pricing", r"aggregation", r"transport", r"selling"])
        judge = _has_any(a, [r"judge", r"measure", r"metric", r"success", r"indicator", r"compare"])
        if (frugal or has_time) and market and judge and (has_gate or has_artifact):
            return 3, 0.72, "test+metrics+decision"
        if (frugal or has_time) and (market or has_gender) and judge:
            return 2, 0.66, "test+metrics"
        if frugal or judge:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    # =====================
    # EP (A3)
    # =====================
    if qid == "EP_Q1":
        spine = _has_any(a, [r"supply\-?chain", r"meal", r"community liaison", r"escalat", r"owner"])
        owners = has_owner
        milestones = _has_any(a, [r"milestone", r"by \w+", r"week", r"month", r"gate"])
        if spine and owners and milestones and (has_gate or has_artifact):
            return 3, 0.75, "execution spine + owners + milestones"
        if spine and owners:
            return 2, 0.70, "spine + owners"
        if spine or owners:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "EP_Q2":
        plan = _has_any(a, [r"90\-?day", r"outcome", r"milestone", r"decision gate", r"gate"])
        owners = has_owner
        slippage = _has_any(a, [r"slippage", r"expose", r"early warning", r"red flag", r"escalat"])
        if plan and owners and slippage and has_gate:
            return 3, 0.75, "90-day plan + gates + early warning"
        if plan and owners:
            return 2, 0.70, "plan + owners"
        if plan or owners:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "EP_Q3":
        handoff = _has_any(a, [r"handoff", r"handover", r"between partners", r"failure", r"gap"])
        artifact = has_artifact or _has_any(a, [r"checklist", r"raci", r"sop", r"template", r"decision log"])
        ritual = _has_any(a, [r"ritual", r"stand\-?up", r"weekly review", r"joint review", r"cadence"])
        if handoff and artifact and ritual:
            return 3, 0.72, "handoff + artifact + ritual"
        if handoff and (artifact or ritual):
            return 2, 0.66, "handoff + mitigation"
        if handoff:
            return 1, 0.60, "handoff mentioned only"
        return 1, 0.55, "generic"

    if qid == "EP_Q4":
        drop = _has_any(a, [r"drop", r"stop", r"remove", r"de\-?prioritize"])
        keep = _has_any(a, [r"keep", r"stays", r"non\-?negotiable", r"must remain"])
        why = _has_any(a, [r"because", r"so that", r"in order to"])
        if drop and keep and why and (has_local or has_gender):
            return 3, 0.70, "clear prioritization + rationale (locally anchored)"
        if drop and keep and why:
            return 2, 0.65, "prioritization + rationale"
        if drop or keep:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    # =====================
    # CFC (A4)
    # =====================
    if qid == "CFC_Q1":
        workshop = _has_any(a, [r"workshop", r"one\-?day", r"agenda", r"session", r"block"])
        outputs = _has_any(a, [r"outputs?", r"deliverables?", r"decisions?", r"action plan", r"raci", r"alignment"])
        if workshop and outputs and (has_artifact or has_owner):
            return 3, 0.70, "agenda + outputs + ownership"
        if workshop and outputs:
            return 2, 0.65, "agenda + outputs"
        if workshop:
            return 1, 0.60, "workshop mentioned only"
        return 1, 0.55, "generic"

    if qid == "CFC_Q2":
        tension = _has_any(a, [r"meal", r"gender", r"tension", r"trade\-?off"])
        integrate = _has_any(a, [r"integrat", r"single system", r"shared", r"one framework"])
        if tension and integrate and safeguards:
            return 3, 0.72, "integrated system + safeguards"
        if tension and integrate:
            return 2, 0.66, "integrated approach"
        if tension:
            return 1, 0.60, "mentions tension only"
        return 1, 0.55, "generic"

    if qid == "CFC_Q3":
        principles = _count_any(a, [r"principle", r"we will", r"must", r"always"]) >= 1
        tests = _has_any(a, [r"test", r"adherence", r"audit", r"review", r"spot check", r"scorecard", r"indicator"])
        if principles and tests and (has_owner or has_artifact):
            return 3, 0.70, "principles + adherence tests"
        if principles and tests:
            return 2, 0.65, "principles + tests"
        if principles:
            return 1, 0.60, "principles only"
        return 1, 0.55, "generic"

    if qid == "CFC_Q4":
        coown = _has_any(a, [r"co\-?own", r"joint decision", r"shared decision", r"not approved"])
        structure = _has_any(a, [r"forum", r"committee", r"decision rights", r"sign\-?off", r"cadence"])
        if coown and structure and safeguards:
            return 3, 0.70, "co-owned + structure + safeguards"
        if coown and structure:
            return 2, 0.65, "co-owned + structure"
        if coown or structure:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    # =====================
    # FTD (A5)
    # =====================
    if qid == "FTD_Q1":
        promise = _has_any(a, [r"promise", r"commit", r"publish", r"weekly"])
        enforce = _has_any(a, [r"enforce", r"accountability", r"consequence", r"escalat", r"owner"])
        if promise and enforce and (has_artifact or has_gate):
            return 3, 0.70, "weekly promise + enforcement"
        if promise and enforce:
            return 2, 0.65, "promise + enforcement"
        if promise:
            return 1, 0.60, "promise only"
        return 1, 0.55, "generic"

    if qid == "FTD_Q2":
        indicators = _has_any(a, [r"indicator", r"metric", r"leading", r"dashboard"])
        owners = has_owner
        escalation = _has_any(a, [r"escalat", r"if .* red", r"turns red", r"rule"])
        if indicators and owners and escalation:
            return 3, 0.70, "dashboard + owners + escalation"
        if indicators and escalation:
            return 2, 0.65, "indicators + escalation"
        if indicators:
            return 1, 0.60, "indicators only"
        return 1, 0.55, "generic"

    if qid == "FTD_Q3":
        miss = _has_any(a, [r"miss", r"two milestones", r"late", r"slip"])
        convo = _has_any(a, [r"conversation", r"reset", r"recovery", r"root cause", r"renegotiate"])
        options = _has_any(a, [r"options?", r"terminate", r"pause", r"replace", r"support", r"penalty"])
        if miss and convo and options and has_gate:
            return 3, 0.70, "recovery conversation + decision options"
        if miss and convo and options:
            return 2, 0.65, "convo + options"
        if miss and (convo or options):
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "FTD_Q4":
        theatre = _has_any(a, [r"update theatre", r"no updates", r"decisional", r"decision log"])
        practices = _has_any(a, [r"pre\-?read", r"action log", r"owner", r"due date", r"timebox"])
        if theatre and practices:
            return 3, 0.68, "decisional meeting system"
        if theatre or practices:
            return 2, 0.62, "anti-theatre practices"
        return 1, 0.55, "generic"

    # =====================
    # LDA (A6)
    # =====================
    if qid == "LDA_Q1":
        pause = _has_any(a, [r"pause", r"reflect", r"quarterly", r"review"])
        questions = _has_any(a, [r"questions?", r"we will ask", r"what worked", r"what didn't", r"why"])
        signed = _has_any(a, [r"signed changes", r"change log", r"update", r"version", r"agre(e|ed)"])
        if pause and questions and signed and safeguards:
            return 3, 0.72, "pause+reflect -> signed changes"
        if pause and questions and signed:
            return 2, 0.66, "pause+signed changes"
        if pause and questions:
            return 1, 0.60, "reflect only"
        return 1, 0.55, "generic"

    if qid == "LDA_Q2":
        hypo = _has_any(a, [r"hypothesis", r"i believe", r"we assume"])
        evidence = _has_any(a, [r"evidence", r"data", r"would change my mind", r"if .* then"])
        nextday = _has_any(a, [r"day after", r"immediately", r"then i would", r"we will pivot", r"adjust"])
        if hypo and evidence and nextday and has_gate:
            return 3, 0.72, "hypothesis + threshold + action"
        if hypo and evidence and nextday:
            return 2, 0.66, "hypothesis + evidence + action"
        if hypo or evidence:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "LDA_Q3":
        negative = _has_any(a, [r"negative finding", r"stall", r"not working", r"harm", r"risk"])
        trust = _has_any(a, [r"trust", r"protect", r"transparent", r"communicat", r"no blame"])
        action = _has_any(a, [r"adapt", r"change", r"mitigate", r"plan", r"support"])
        if negative and trust and action and safeguards:
            return 3, 0.70, "negative finding + trust + adaptation"
        if negative and trust and action:
            return 2, 0.64, "handle negative finding well"
        if negative and (trust or action):
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "LDA_Q4":
        stop = _has_any(a, [r"stop", r"drop", r"cut", r"pause"])
        fund = _has_any(a, [r"fund", r"reallocate", r"budget", r"free up"])
        adapt = _has_any(a, [r"adapt", r"adjust", r"change", r"iteration"])
        if stop and fund and adapt and (has_tradeoff or has_gate):
            return 3, 0.68, "stop + reallocate + adaptations"
        if stop and fund:
            return 2, 0.62, "stop + reallocate"
        if stop:
            return 1, 0.60, "stop only"
        return 1, 0.55, "generic"

    # =====================
    # RDM (A7)
    # =====================
    if qid == "RDM_Q1":
        compare = _has_any(a, [r"cost", r"benefit", r"risk", r"equity", r"implication", r"side\-by\-side", r"trade\-?off"])
        call = _has_any(a, [r"my call", r"i would choose", r"decision", r"therefore", r"we will"])
        if compare and call and (has_gate or has_time):
            return 3, 0.72, "structured comparison + clear call"
        if compare and call:
            return 2, 0.66, "comparison + call"
        if call or compare:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "RDM_Q2":
        data_pts = _count_any(a, [r"data", r"evidence", r"from", r"need", r"two"]) >= 1
        from_who = _has_any(a, [r"from \w+", r"ministry", r"partners?", r"women", r"district", r"kujenga"])
        fallback = _has_any(a, [r"fallback", r"if .* don't", r"if .* not arrive", r"assume", r"proceed with"])
        if (has_time or _has_any(a,[r"by friday"])) and data_pts and from_who and fallback:
            return 3, 0.70, "data needs + source + fallback under deadline"
        if data_pts and from_who and fallback:
            return 2, 0.64, "data + source + fallback"
        if data_pts or fallback:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "RDM_Q3":
        socialize = _has_any(a, [r"socializ", r"communicat", r"explain", r"share rationale", r"transparen"])
        respect = _has_any(a, [r"respect", r"listen", r"partners feel", r"even if", r"disagree"])
        process = _has_any(a, [r"forum", r"meeting", r"brief", r"memo", r"consult"])
        if socialize and respect and process and safeguards:
            return 3, 0.70, "transparent socialization + respect + process"
        if socialize and respect:
            return 2, 0.64, "socialize trade-off well"
        if socialize or respect:
            return 1, 0.60, "partial"
        return 1, 0.55, "generic"

    if qid == "RDM_Q4":
        rule = _has_any(a, [r"decision rule", r"rule", r"if regions", r"when regions", r"coherence", r"standard"])
        localflex = _has_any(a, [r"local initiative", r"adapt", r"flex", r"context"])
        if rule and localflex and safeguards:
            return 3, 0.70, "rule + local flex + safeguards"
        if rule and localflex:
            return 2, 0.64, "rule + local flex"
        if rule:
            return 1, 0.60, "rule only"
        return 1, 0.55, "generic"

    # Fallback: if it has multiple strong evidence signals, treat as 2
    evidence_hits = sum([has_time, has_owner, has_artifact, has_gate, has_tradeoff, has_local, has_gender])
    if evidence_hits >= 4:
        return 2, 0.60, "fallback(evidence-rich)"
    if evidence_hits <= 1:
        return 1, 0.55, "fallback(generic)"
    return 1, 0.58, "fallback"



# ==============================
# KOBO
# ==============================
def kobo_url(asset_uid: str, kind: str = "submissions"):
    return f"{KOBO_BASE.rstrip('/')}/api/v2/assets/{asset_uid}/{kind}/?format=json"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_dataframe() -> pd.DataFrame:
    if not KOBO_ASSET_ID1 or not KOBO_TOKEN:
        st.warning("Set KOBO_ASSET_ID1 and KOBO_TOKEN in st.secrets.")
        return pd.DataFrame()
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    for kind in ("submissions","data"):
        url = kobo_url(KOBO_ASSET_ID1, kind)
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
            if r.status_code in (401,403):
                st.error("Kobo auth failed: check KOBO_TOKEN and tenant.")
                return pd.DataFrame()
            st.error(f"Kobo error {r.status_code}: {r.text[:300]}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch Kobo data: {e}")
            return pd.DataFrame()
    st.error("Could not fetch Kobo data. Check KOBO_BASE/ASSET/TOKEN.")
    return pd.DataFrame()


# ==============================
# MAPPING + EXEMPLARS
# ==============================
QID_PREFIX_TO_SECTION = {"LAV":"A1","II":"A2","EP":"A3","CFC":"A4","FTD":"A5","LDA":"A6","RDM":"A7"}
QNUM_RX = re.compile(r"_Q(\d+)$")

def load_mapping_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    m = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    m.columns = [c.lower().strip() for c in m.columns]
    assert {"column","question_id","attribute"}.issubset(m.columns), "mapping needs: column, question_id, attribute"
    if "prompt_hint" not in m.columns:
        m["prompt_hint"] = ""

    norm = lambda s: re.sub(r"\s+"," ", str(s).strip().lower())
    target = {norm(a): a for a in ORDERED_ATTRS}

    def snap_attr(a):
        key = norm(a)
        if key in target:
            return target[key]
        best = process.extractOne(key, list(target.keys()), scorer=fuzz.token_set_ratio)
        return target[best[0]] if best and best[1] >= 75 else None

    m["attribute"] = m["attribute"].apply(snap_attr)
    m = m[m["attribute"].notna()].copy()
    return m

def read_jsonl_path(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exemplars file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def build_kobo_base_from_qid(question_id: str) -> list[str] | None:
    if not question_id:
        return None
    qid = question_id.strip().upper()
    m = QNUM_RX.search(qid)
    if not m:
        return None
    qn = m.group(1)
    prefix = qid.split("_Q")[0]
    sect = QID_PREFIX_TO_SECTION.get(prefix)
    if not sect:
        return None
    token = f"{sect}_{qn}"
    roots = ["Thought Leadership", "Leadership"]
    return [f"{root}/{sect}_Section/{token}" for root in roots]

def expand_possible_kobo_columns(base: str) -> list[str]:
    if not base:
        return []
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
    if c == t:
        return 100
    s = 0
    if c.endswith("/"+t): s = max(s,95)
    if f"/{t}/" in c: s = max(s,92)
    if f"/{t} " in c or f"{t} :: " in c or f"{t} - " in c or f"{t}_" in c: s = max(s,90)
    if t in c: s = max(s,80)
    if "english" in c or "label" in c or "(en)" in c: s += 5
    if "answer (text)" in c or "answer_text" in c or "text" in c: s += 5
    if "thought leadership/" in c or "leadership/" in c or "/a" in c: s += 2
    return s

def resolve_kobo_column_for_mapping(df_cols: list[str], question_id: str, prompt_hint: str) -> str | None:
    bases = build_kobo_base_from_qid(question_id) or []
    variants = []
    for base in bases:
        variants.extend(expand_possible_kobo_columns(base))
    for v in variants:
        if v in df_cols:
            return v
    for c in df_cols:
        if any(c.startswith(b) for b in bases):
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
        best, bs = None, 0
        for c in df_cols:
            sc = _score_kobo_header(c, token)
            if sc > bs:
                bs, best = sc, c
        if best and bs >= 82:
            return best

    hint = clean(prompt_hint or "")
    if hint:
        hits = process.extract(hint, df_cols, scorer=fuzz.partial_token_set_ratio, limit=5)
        for col, score, _ in hits:
            if score >= 80:
                return col

    return None


# ==============================
# EMBEDDINGS (batched, cached)
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def _embed_texts_cached(texts_tuple: tuple[str, ...]) -> dict:
    texts = list(texts_tuple)
    embs = get_embedder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return {t: e for t, e in zip(texts, embs)}

_EMB_CACHE: dict[str, np.ndarray] = {}

def embed_many(texts: list[str]) -> None:
    missing = [t for t in texts if t and t not in _EMB_CACHE]
    if not missing:
        return
    pack = _embed_texts_cached(tuple(missing))
    _EMB_CACHE.update(pack)

def emb_of(text: str):
    t = clean(text)
    return _EMB_CACHE.get(t, None)


# ==============================
# NEW: EXEMPLAR PACKS (replaces centroids)
# ==============================
def _build_pack(texts: list[str], scores: list[int]):
    # de-dup exact pairs but keep order
    seen = set()
    tt, ss = [], []
    for t, s in zip(texts, scores):
        t = clean(t)
        if not t:
            continue
        key = (t, int(s))
        if key in seen:
            continue
        seen.add(key)
        tt.append(t)
        ss.append(int(s))

    embed_many(list(set(tt)))

    vecs, keep_t, keep_s = [], [], []
    for t, s in zip(tt, ss):
        v = emb_of(t)
        if v is None:
            continue
        vecs.append(v)
        keep_t.append(t)
        keep_s.append(int(s))

    if not vecs:
        mat = np.zeros((0, 384), dtype=np.float32)
    else:
        mat = np.vstack(vecs)

    return {"vecs": mat, "scores": np.array(keep_s, dtype=int), "texts": keep_t}

def build_exemplar_packs(exemplars: list[dict]):
    by_qkey, by_attr, question_texts = {}, {}, []
    all_texts = []

    for e in exemplars:
        qid   = clean(e.get("question_id",""))
        qtext = clean(e.get("question_text",""))
        txt   = clean(e.get("text",""))
        attr  = clean(e.get("attribute",""))
        try:
            sc = int(e.get("score", 0))
        except Exception:
            sc = 0

        if not (qid or qtext) or not txt:
            continue

        key = qid if qid else qtext
        pack = by_qkey.setdefault(key, {"question_text": qtext, "scores": [], "texts": [], "attribute": attr})
        pack["scores"].append(sc)
        pack["texts"].append(txt)
        if qtext:
            question_texts.append(qtext)

        by_attr.setdefault(attr, {"scores": [], "texts": []})
        by_attr[attr]["scores"].append(sc)
        by_attr[attr]["texts"].append(txt)

        all_texts.append(txt)

    embed_many(list(set(all_texts)))

    q_packs = {k: _build_pack(v["texts"], v["scores"]) for k, v in by_qkey.items()}
    a_packs = {a: _build_pack(v["texts"], v["scores"]) for a, v in by_attr.items()}

    # global
    g_txt, g_sc = [], []
    for e in exemplars:
        t = clean(e.get("text",""))
        if not t:
            continue
        try:
            s = int(e.get("score", 0))
        except Exception:
            s = 0
        g_txt.append(t); g_sc.append(s)
    g_pack = _build_pack(g_txt, g_sc)

    # de-dup question_texts
    seen=set(); question_texts=[x for x in question_texts if not (x in seen or seen.add(x))]

    return q_packs, a_packs, g_pack, by_qkey, question_texts

def resolve_qkey(q_packs, by_qkey, question_texts, qid: str, prompt_hint: str):
    qid = (qid or "").strip()
    if qid and qid in q_packs:
        return qid
    hint = clean(prompt_hint or "")
    if not (hint and question_texts):
        return None
    match = process.extractOne(hint, question_texts, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        wanted = match[0]
        for k, pack in by_qkey.items():
            if clean(pack.get("question_text","")) == wanted:
                return k
    return None

def score_vec_against_pack(pack: dict, vec: np.ndarray):
    if pack is None or pack["vecs"].size == 0 or vec is None:
        return None, 0.0

    sims = pack["vecs"] @ vec  # cosine similarity (normalized embeddings)
    if sims.size == 0:
        return None, 0.0

    # top1: score of nearest exemplar
    if KNN_METHOD == "top1":
        i = int(np.argmax(sims))
        return int(pack["scores"][i]), float(sims[i])

    # softmax vote over top-k
    k = max(1, min(KNN_K, sims.size))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    top_sims = sims[idx]
    top_scores = pack["scores"][idx]

    w = np.exp((top_sims - top_sims.max()) / float(KNN_TEMP))
    w = w / (w.sum() + 1e-9)
    class_w = np.zeros(4, dtype=float)
    for s, wi in zip(top_scores, w):
        if 0 <= int(s) <= 3:
            class_w[int(s)] += float(wi)
    pred = int(class_w.argmax())
    conf = float(class_w.max())
    return pred, conf


# ==============================
# SCORING (keeps your working table layout)
# ==============================
def score_dataframe(df: pd.DataFrame, mapping: pd.DataFrame,
                    q_packs, a_packs, g_pack,
                    by_qkey, question_texts):

    df_cols = list(df.columns)

    def want_col(c):
        lc = c.strip().lower()
        return any(h in lc for h in PASSTHROUGH_HINTS)
    passthrough_cols = [c for c in df_cols if want_col(c)]
    seen = set()
    passthrough_cols = [x for x in passthrough_cols if not (x in seen or seen.add(x))]

    staff_id_col   = next((c for c in df.columns if c.strip().lower() in ("staff id","staff_id","staffid")), None)
    care_staff_col = next((c for c in df.columns if c.strip().lower() in ("care_staff","care staff","care-staff")), None)

    date_cols_pref = [
        "_submission_time","SubmissionDate","submissiondate",
        "end","End","start","Start","today","date","Date"
    ]
    date_col = next((c for c in date_cols_pref if c in df.columns), df.columns[0])

    start_col = next((c for c in ["start"] if c in df.columns), None)
    end_col   = next((c for c in ["end"] if c in df.columns), None)

    n_rows = len(df)

    # Clean Date column
    if date_col in df.columns:
        date_clean = (
            df[date_col].astype(str).str.strip().str.lstrip(",")
        )
        dt_series = pd.to_datetime(date_clean, errors="coerce")
    else:
        dt_series = pd.Series([pd.NaT] * n_rows)

    # start/end to compute duration
    if start_col:
        start_clean = df[start_col].astype(str).str.strip().str.lstrip(",")
        start_dt = pd.to_datetime(start_clean, utc=True, errors="coerce")
    else:
        start_dt = pd.Series([pd.NaT] * n_rows)

    if end_col:
        end_clean = df[end_col].astype(str).str.strip().str.lstrip(",")
        end_dt = pd.to_datetime(end_clean, utc=True, errors="coerce")
    else:
        end_dt = pd.Series([pd.NaT] * n_rows)

    duration_min = (end_dt - start_dt).dt.total_seconds() / 60.0
    duration_min = duration_min.clip(lower=0)

    # mapping resolution (KEEP YOUR WORKING LOGIC)
    all_mapping = [r for r in mapping.to_dict(orient="records") if r["attribute"] in ORDERED_ATTRS]
    resolved_for_qid = {}
    for r in all_mapping:
        qid   = r["question_id"]
        qhint = r.get("prompt_hint","")
        hit = resolve_kobo_column_for_mapping(df_cols, qid, qhint)
        if hit:
            resolved_for_qid[qid] = hit

    # batch-embed distinct answers
    distinct_by_qid: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        for r in all_mapping:
            qid = r["question_id"]
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                a = clean(row.get(col, ""))
                if a:
                    distinct_by_qid.setdefault(qid, set()).add(a)

    all_distinct = set()
    for s in distinct_by_qid.values():
        all_distinct |= s
    embed_many(list(all_distinct))

    exact_sc_cache: dict[tuple[str, str], int] = {}
    dup_bank: dict[str, list[tuple[np.ndarray, int]]] = {}

    out_rows = []
    for i, resp in df.iterrows():
        row = {}

        row["Date"] = (pd.to_datetime(dt_series.iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
                       if pd.notna(dt_series.iloc[i]) else str(i))
        val = duration_min.iloc[i]
        row["Duration"] = int(round(val)) if not pd.isna(val) else ""

        who_col = care_staff_col or staff_id_col
        row["Care_Staff"] = str(resp.get(who_col)) if who_col else ""

        for c in passthrough_cols:
            lc = c.strip().lower()
            if lc in _EXCLUDE_SOURCE_COLS_LOWER:
                continue
            if c in ("Date","Duration","Care_Staff"):
                continue
            row[c] = resp.get(c, "")

        per_attr = {}
        any_ai = False
        qtext_cache = {}

        override_count = 0
        needs_review = False

        row_answers = {}
        for r in all_mapping:
            qid = r["question_id"]
            col = resolved_for_qid.get(qid)
            if col and col in df.columns:
                row_answers[qid] = clean(resp.get(col, ""))

        for r in all_mapping:
            qid, attr, qhint = r["question_id"], r["attribute"], r.get("prompt_hint","")
            col = resolved_for_qid.get(qid)
            if not col:
                continue
            ans = row_answers.get(qid, "")
            if not ans:
                continue
            vec = emb_of(ans)
            ex_conf = None
            ex_score = None

            qkey = resolve_qkey(q_packs, by_qkey, question_texts, qid, qhint)
            if qkey and qkey not in qtext_cache:
                qtext_cache[qkey] = (by_qkey.get(qkey, {}) or {}).get("question_text","")
            qhint_full = qtext_cache.get(qkey, "") if qkey else qhint

            was_cached = (qid, ans) in exact_sc_cache
            sc = exact_sc_cache.get((qid, ans))
            reused = False

            # duplicate reuse
            if sc is None and vec is not None:
                best_dup_sc, best_dup_sim = None, -1.0
                for v2, sc2 in dup_bank.get(qid, []):
                    sim = float(np.dot(vec, v2))
                    if sim > best_dup_sim:
                        best_dup_sim, best_dup_sc = sim, sc2
                if best_dup_sc is not None and best_dup_sim >= DUP_SIM:
                    sc = int(best_dup_sc)
                    reused = True

            # NEW scoring: nearest exemplar (question -> attribute -> global)
            if sc is None and vec is not None:
                sc2, conf = None, 0.0

                if qkey and qkey in q_packs:
                    sc2, conf = score_vec_against_pack(q_packs[qkey], vec)

                if sc2 is None and attr in a_packs:
                    sc2, conf = score_vec_against_pack(a_packs[attr], vec)

                if sc2 is None:
                    sc2, conf = score_vec_against_pack(g_pack, vec)

                sc = sc2
                ex_score = sc2
                ex_conf = conf

                # optional low-confidence clamp (disabled by default)
                if sc is not None and CONF_CLAMP > 0 and conf < CONF_CLAMP:
                    sc = min(int(sc), 1)

                # off-topic clamp (keep your old behavior)
                if sc is not None and qa_overlap(ans, qhint_full or qhint) < MIN_QA_OVERLAP:
                    sc = min(int(sc), 1)


            # ------------------------------
            # Rubric validator + override (post-score)
            # ------------------------------
            # This runs even if exemplar score came from cache/dup reuse.
            # Policy (safe defaults):
            # - upgrade 0/1 -> 2/3 when rubric is confident and evidence is present
            # - downgrade only for empty/harmful (or if RUBRIC_DOWNGRADE=True)
            rub_sc, rub_conf, rub_reason = rubric_validate(qid, ans)
            final_sc = sc

            if RUBRIC_OVERRIDE:
                # hard downgrades
                if rub_reason in ("empty", "harmful/derailing") and rub_conf >= 0.85:
                    if final_sc is None:
                        final_sc = rub_sc
                    else:
                        final_sc = min(int(final_sc), int(rub_sc))
                    needs_review = True

                # upgrades: common failure mode where exemplars under-score concise but correct answers
                if final_sc is not None:
                    if int(final_sc) <= 1 and int(rub_sc) >= 2 and rub_conf >= RUBRIC_UPGRADE_CONF:
                        # only upgrade when answer isn't extremely short
                        if len(ans) >= 18:
                            if int(rub_sc) != int(final_sc):
                                override_count += 1
                            final_sc = int(rub_sc)

                    # optional soft downgrades for overly short/generic answers
                    if RUBRIC_DOWNGRADE and int(rub_sc) <= 1 and rub_conf >= 0.85:
                        if len(ans) < 25:
                            if int(rub_sc) != int(final_sc):
                                override_count += 1
                            final_sc = int(rub_sc)

            if final_sc is not None:
                sc = max(0, min(3, int(final_sc)))

            # Optional audit columns (useful while tuning)
            if RUBRIC_SHOW_AUDIT:
                row[f"{attr}_ExemplarScore_{qid}"] = "" if ex_score is None else int(ex_score)
                row[f"{attr}_ExemplarConf_{qid}"]  = "" if ex_conf is None else round(float(ex_conf), 3)
                row[f"{attr}_RubricScore_{qid}"]   = int(rub_sc)
                row[f"{attr}_RubricConf_{qid}"]    = round(float(rub_conf), 2)
                row[f"{attr}_RubricReason_{qid}"]  = rub_reason


            if sc is not None and not was_cached:
                exact_sc_cache[(qid, ans)] = int(sc)
                if vec is not None and not reused:
                    bank = dup_bank.setdefault(qid, [])
                    if len(bank) < 300:
                        bank.append((vec, int(sc)))

            ai_score = ai_signal_score(ans, qhint_full)
            if ai_score >= AI_SUSPECT_THRESHOLD:
                any_ai = True

            qn = None
            if "_Q" in (qid or ""):
                try:
                    qn = int(qid.split("_Q")[-1])
                except Exception:
                    qn = None

            if qn in (1,2,3,4) and sc is not None:
                sk = f"{attr}_Qn{qn}"
                rk = f"{attr}_Rubric_Qn{qn}"
                row[sk] = int(sc)
                row[rk] = BANDS[int(sc)]
                per_attr.setdefault(attr, []).append(int(sc))

        # fill blanks
        for attr in ORDERED_ATTRS:
            for qn in (1,2,3,4):
                row.setdefault(f"{attr}_Qn{qn}", "")
                row.setdefault(f"{attr}_Rubric_Qn{qn}", "")

        # attribute averages + ranks (keep your original rounding)
        overall_total = 0
        for attr in ORDERED_ATTRS:
            scores = per_attr.get(attr, [])
            if not scores:
                row[f"{attr}_Avg (0–3)"] = ""
                row[f"{attr}_RANK"]      = ""
            else:
                avg = float(np.mean(scores))
                band = int(round(avg))
                overall_total += band
                row[f"{attr}_Avg (0–3)"] = round(avg, 2)
                row[f"{attr}_RANK"]      = BANDS[band]

        row["Overall Total (0–21)"] = overall_total
        row["Overall Rank"] = next((label for (label, lo, hi) in OVERALL_BANDS if lo <= overall_total <= hi), "")
        row["Overrides_Applied"] = int(override_count)
        row["Needs_Review"] = bool(needs_review)
        row["AI_suspected"] = bool(any_ai)
        out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # column order (keep your working layout)
    ordered = [c for c in ["Date","Duration","Care_Staff"] if c in res.columns]
    source_cols = [c for c in df.columns if c.strip().lower() not in _EXCLUDE_SOURCE_COLS_LOWER]
    source_cols = [c for c in source_cols if c not in ("Date","Duration","Care_Staff")]
    ordered += [c for c in source_cols if c in res.columns]

    mid_q = []
    for attr in ORDERED_ATTRS:
        for qn in (1,2,3,4):
            mid_q += [f"{attr}_Qn{qn}", f"{attr}_Rubric_Qn{qn}"]
    ordered += [c for c in mid_q if c in res.columns]

    mid_a = []
    for attr in ORDERED_ATTRS:
        mid_a += [f"{attr}_Avg (0–3)", f"{attr}_RANK"]
    ordered += [c for c in mid_a if c in res.columns]

    ordered += [c for c in ["Overall Total (0–21)","Overall Rank"] if c in res.columns]
    # quality controls
    for c in ["Overrides_Applied","Needs_Review"]:
        if c in res.columns:
            ordered += [c]
    if "AI_suspected" in res.columns:
        ordered += ["AI_suspected"]

    res = res.reindex(columns=ordered)
    return res


# ==============================
# EXPORTS / SHEETS
# ==============================
def _ensure_ai_last(df: pd.DataFrame,
                    export_name: str = "AI_Suspected",
                    source_name: str = "AI_suspected") -> pd.DataFrame:
    out = df.copy()
    if export_name not in out.columns:
        if source_name in out.columns:
            out = out.rename(columns={source_name: export_name})
        else:
            out[export_name] = ""
    cols = [c for c in out.columns if c != export_name] + [export_name]
    return out[cols]

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    df_out = _ensure_ai_last(df)
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df_out.to_excel(w, index=False)
    return bio.getvalue()


SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
DEFAULT_WS_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME1", "Thought Leadership")

def _normalize_sa_dict(raw: dict) -> dict:
    if not raw:
        raise ValueError("gcp_service_account missing in secrets.")
    sa = dict(raw)
    if "token_ur" in sa and "token_uri" not in sa:
        sa["token_uri"] = sa.pop("token_ur")
    if sa.get("private_key") and "\\n" in sa["private_key"]:
        sa["private_key"] = sa["private_key"].replace("\\n","\n")
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
    sh = gc.open_by_key(key)
    try:
        return sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=ws_name, rows="20000", cols="150")

def _post_write_formatting(ws: gspread.Worksheet, cols: int) -> None:
    try:
        ws.freeze(rows=1)
    except Exception:
        pass
    try:
        ws.spreadsheet.batch_update({
            "requests":[{"autoResizeDimensions":{
                "dimensions":{"sheetId": ws.id, "dimension":"COLUMNS","startIndex":0,"endIndex":cols}
            }}]
        })
    except Exception:
        pass

def upload_df_to_gsheets(df: pd.DataFrame) -> tuple[bool, str]:
    try:
        ws = _open_ws_by_key()
        df_out = _ensure_ai_last(df, export_name="AI_Suspected", source_name="AI_suspected")
        header = df_out.columns.astype(str).tolist()
        values = df_out.astype(object).where(pd.notna(df_out), "").values.tolist()
        ws.clear()
        ws.spreadsheet.values_batch_update(
            body={"valueInputOption":"USER_ENTERED","data":[{"range": f"'{ws.title}'!A1","values":[header]+values}]}
        )
        _post_write_formatting(ws, len(header))
        return True, f"✅ Wrote {len(values)} rows × {len(header)} cols to '{ws.title}' (last='AI_Suspected')."
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"


# ==============================
# MAIN
# ==============================
def main():
    inject_css()

    st.markdown("""
        <div class="app-header-card">
            <div class="pill">Thought Leadership • Auto Scoring</div>
            <h1>Thought Leadership</h1>
            <p class="app-header-subtitle">
                Importing Kobo submissions, scoring CARE thought leadership attributes (nearest exemplars),
                flagging AI-like responses, and exporting results to Google Sheets.
            </p>
        </div>
    """, unsafe_allow_html=True)

    try:
        mapping = load_mapping_from_path(MAPPING_PATH)
    except Exception as e:
        st.error(f"Failed to load mapping: {e}")
        return

    try:
        exemplars = read_jsonl_path(EXEMPLARS_PATH)
        if not exemplars:
            st.error("Exemplars file is empty.")
            return
    except Exception as e:
        st.error(f"Failed to read exemplars: {e}")
        return

    with st.spinner("Building exemplar packs (nearest-neighbour scorer)..."):
        q_packs, a_packs, g_pack, by_q, qtexts = build_exemplar_packs(exemplars)

    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.warning("No Kobo submissions found.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Fetched dataset")
    st.caption(f"Rows: {len(df):,}  •  Columns: {len(df.columns):,}")
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Scoring (+ AI detection)..."):
        scored = score_dataframe(df, mapping, q_packs, a_packs, g_pack, by_q, qtexts)

    st.success("✅ Scoring complete.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Scored table")
    st.caption("Same output layout as your working version; scoring now uses nearest exemplars.")
    styled = scored.style.apply(
        lambda r: ["background-color: #241E4E"] * len(r) if ("AI_suspected" in r and r["AI_suspected"]) else ["" for _ in r],
        axis=1
    )
    st.dataframe(styled, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⬇️ Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Excel",
            data=to_excel_bytes(scored),
            file_name="Leadership_Scoring.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=_ensure_ai_last(scored).to_csv(index=False).encode("utf-8"),
            file_name="Leadership_Scoring.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if AUTO_PUSH:
        with st.spinner("📤 Sending to Google Sheets..."):
            ok, msg = upload_df_to_gsheets(scored)
        (st.success if ok else st.error)(msg)

if __name__ == "__main__":
    main()
