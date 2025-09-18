import streamlit as st
import requests
import pandas as pd
import time
from transformers import pipeline
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# --------------------------
# 1️⃣ CONFIG
# --------------------------
KOBO_FORM_ID = "atdspJQv7RBwjkmaVFRS43"
KOBO_API_URL = f"https://kf.kobotoolbox.org/assets/atdspJQv7RBwjkmaVFRS43/submissions/?format=json"

KOBO_TOKEN = st.secrets["KOBO_TOKEN"]
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}


# 2️⃣ SECTION MAP & RUBRICS
# --------------------------
SECTION_MAP = {
    "case1_stratpos_group": "Influencing",
    "case1_stakeholder_group": "Stakeholder Mapping & Engagement",
    "case1_evidence_group": "Evidence-Informed Advocacy",
    "case1_comm_group": "Communication, Framing & Messaging",
    "case1_risk_group": "Risk Awareness & Mitigation",
    "case1_coalition_group": "Coalition Building & Collaborative Action",
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
# 5️⃣ LOCAL AI SCORING
@st.cache_resource
def load_ml_models():
    # Load your score prediction model
    clf = joblib.load("score_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    # Optional: Load theme extraction model
    theme_clf = joblib.load("theme_model.pkl")
    theme_vectorizer = joblib.load("theme_vectorizer.pkl")
    return clf, vectorizer, theme_clf, theme_vectorizer

clf, vectorizer, theme_clf, theme_vectorizer = load_ml_models()

# --------------------------
# 6️⃣ ML SCORING AND THEME EXTRACTION
# --------------------------
def ml_score_and_theme(answers):
    # Predict scores
    X = vectorizer.transform(answers)
    scores = clf.predict(X)
    # Predict themes
    X_theme = theme_vectorizer.transform(answers)
    themes = theme_clf.predict(X_theme)
    return scores, themes

# --------------------------
# 7️⃣ STREAMLIT APP (SCORING LOOP)
# --------------------------
st.title("📊 Kobo AI Dashboard (Fast ML Scoring)")

df = fetch_kobo_data()
if not df.empty:
    st.subheader("Raw Responses")
    st.dataframe(df.head())

    flat_df = flatten_kobo_responses(df)

    st.subheader("Scoring with ML...")
    batch_size = 50
    scored_list = []

    for i in range(0, len(flat_df), batch_size):
        batch = flat_df.iloc[i:i+batch_size]

        pred_scores, pred_themes = ml_score_and_theme(batch["Answer"].tolist())

        for score, theme, (_, row) in zip(pred_scores, pred_themes, batch.iterrows()):
            section_prefix = "_".join(row["Question_ID"].split("_")[:2]) + "_group"
            section_name = SECTION_MAP.get(section_prefix, section_prefix)

            scored_list.append({
                "Respondent_ID": row["Respondent_ID"],
                "Section": section_name,
                "Question_ID": row["Question_ID"],
                "Answer": row["Answer"],
                "Score": score,
                "Themes": theme
            })

    scored_df = pd.DataFrame(scored_list)
    st.subheader("✅ ML Scored & Themed Responses")
    st.dataframe(scored_df)
