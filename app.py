import streamlit as st
import requests
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import time
# --------------------------
# 1️⃣ CONFIG
# --------------------------
KOBO_FORM_ID = "atdspJQv7RBwjkmaVFRS43"
KOBO_API_URL = f"https://kf.kobotoolbox.org/assets/atdspJQv7RBwjkmaVFRS43/submissions/?format=json"

KOBO_TOKEN = st.secrets["KOBO_TOKEN"]
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}


# 2️⃣ SECTION MAP & RUBRICS
# --------------------------
# Map Question_ID prefixes to Competency / Attribute
SECTION_MAP = {
    "case1_stratpos_group": "Strategic Positioning & Donor Fluency",
    "case1_stakeholder_group": "Power-Aware Stakeholder Mapping",
    "case1_evidence_group": "Evidence-Led Learning",
    "case1_comm_group": "Communication, Framing & Messaging",
    "case1_risk_group": "Risk Awareness & Mitigation",
    "case1_coalition_group": "Coalition Governance & Convening",
    "case1_adaptive_group": "Adaptive Tactics & Channel Selection",
    "case1_integrity_group": "Integrity & Values-Based Influencing",
    
    "case2_strategic_group": "Advisory Skills",
    "case2_credibility_group": "Credibility & Trustworthiness",
    "case2_comm_group": "Effective Communication & Influence",
    "case2_client_group": "Client & Stakeholder Focus",
    "case2_collab_group": "Fostering Collaboration & Partnership",
    "case2_impact_group": "Ensuring Relevance & Impact",
    "case2_solution_group": "Solution Orientation & Adaptability",
    "case2_capacity_group": "Capacity Strengthening & Empowerment Support",
    
    "case3_vision_group": "Strategic Thought Leadership",
    "case3_innovation_group": "Innovation & Insight",
    "case3_execution_group": "Execution Planning",
    "case3_collab_group": "Cross-Functional Collaboration",
    "case3_discipline_group": "Follow-Through Discipline",
    "case3_learning_group": "Learning-Driven Adjustment",
    "case3_results_group": "Result-Oriented Decision-Making",
    
    "case4_positioning_group": "Strategic Positioning & Donor Fluency",
    "case4_stakeholders_group": "Power-Aware Stakeholder Mapping",
    "case4_allyship_group": "Equitable Allyship & Local Fronting",
    "case4_coalition_group": "Coalition Governance & Convening",
    "case4_messaging_group": "Community-Centered Messaging",
    "case4_evidence_group": "Evidence-Led Learning",
    "case4_influence_group": "Influence Without Authority",
    "case4_risk_group": "Risk Management & Adaptive Communication",
    
    "case5_learning_group": "Learning Agility",
    "case5_feedback_group": "Feedback Seeking & Responsiveness",
    "case5_resilience_group": "Resilience & Adaptability",
    "case5_reflect_group": "Reflective Practice & Self-Awareness",
    "case5_innovation_group": "Innovation & Experimentation",
    "case5_context_group": "Contextual Intelligence / Systems Thinking",
}


DEFAULT_RUBRIC = "3 – Transformative, 2 – Strategic, 1 – Compliant, 0 – Counterproductive"

# Optional: customize rubric per section
SECTION_RUBRICS = {sec: DEFAULT_RUBRIC for sec in SECTION_MAP.values()}


# --------------------------
# 2️⃣ FUNCTIONS
# --------------------------
@st.cache_data(ttl=300)
def fetch_kobo_data():
    """Fetch submissions from KoboToolbox and return as a DataFrame."""
    try:
        response = requests.get(KOBO_API_URL, headers=HEADERS)
        

        # Raise an error if the request failed
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

    # Try to parse JSON
    try:
        data = response.json()
    except ValueError:
        st.error(f"Failed to parse JSON. Response text (truncated):\n{response.text[:1000]}")
        return pd.DataFrame()

    # Determine if the data is a list (common for submissions endpoint)
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = data.get("results", [])
    else:
        st.error("Unexpected data format from Kobo API.")
        return pd.DataFrame()

    if not results:
        st.warning("No submissions found yet.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Optional: preview first few rows in Streamlit
    st.write("Preview of fetched data:")
    st.dataframe(df.head())

    return df
# --------------------------
 #3️⃣ FLATTEN RESPONSES
# 4️⃣ FLATTEN KOBO RESPONSES
# --------------------------
def flatten_kobo_responses(df):
    rows = []
    for idx, row in df.iterrows():
        respondent_id = row.get("_id", f"resp_{idx}")
        for col in df.columns:
            if col.startswith("case"):
                answer = row[col]
                if pd.notna(answer) and answer.strip() != "":
                    rows.append({
                        "Respondent_ID": respondent_id,
                        "Question_ID": col,
                        "Answer": answer
                    })
    return pd.DataFrame(rows)

# --------------------------
# --------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

RUBRIC_KEYWORDS = {
    "3 – Transformative": ["innovative", "transformative", "impactful", "change", "improve"],
    "2 – Strategic": ["strategic", "planned", "goal", "objective", "aligned"],
    "1 – Compliant": ["followed instructions", "adhered", "standard", "compliant"],
    "0 – Counterproductive": ["ignored", "failed", "mistake", "problem", "conflict"]
}

def score_answer(answer):
    answer_lower = answer.lower()
    for score, keywords in RUBRIC_KEYWORDS.items():
        if any(k in answer_lower for k in keywords):
            return score
    return "2 – Strategic"  # default

def extract_themes(answer, top_n=3):
    words = preprocess_text(answer).split()
    word_counts = Counter(words)
    common = [w for w, c in word_counts.most_common(top_n)]
    return ", ".join(common)

# --------------------------
# 4️⃣ STREAMLIT APP
# --------------------------
st.title("📊 Kobo Qualitative Analysis Dashboard")

df = fetch_kobo_data()
if df.empty:
    st.warning("No data found.")
else:
    flat_df = flatten_kobo_responses(df)
    flat_df["Score"] = flat_df["Answer"].apply(score_answer)
    flat_df["Themes"] = flat_df["Answer"].apply(extract_themes)

    # Dynamically create section name from question ID prefix
    flat_df["Section"] = flat_df["Question_ID"].apply(lambda qid: "_".join(qid.split("_")[:3]))

    # Final summary: counts per section
    summary_df = flat_df.groupby("Section")["Score"].value_counts().unstack(fill_value=0)
    summary_df = summary_df[["0 – Counterproductive", "1 – Compliant", "2 – Strategic", "3 – Transformative"]].fillna(0)

    st.subheader("✅ Scored & Themed Responses (Full Detail)")
    st.dataframe(flat_df)

    st.subheader("✅ Final Summary by Section")
    st.dataframe(summary_df)
