import streamlit as st
import requests
import pandas as pd
import re
from collections import Counter
import time
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from sentence_transformers import SentenceTransformer, util
# --------------------------
# 1️⃣ CONFIG
# --------------------------
KOBO_FORM_ID = "atdspJQv7RBwjkmaVFRS43"
KOBO_API_URL = f"https://kf.kobotoolbox.org/assets/atdspJQv7RBwjkmaVFRS43/submissions/?format=json"

KOBO_TOKEN = st.secrets["KOBO_TOKEN"]
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}

# Power BI Push Dataset URL
POWERBI_PUSH_URL = st.secrets["POWERBI_PUSH_URL"]  
MAX_LEN = 4000  # Power BI string limit
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


DEFAULT_RUBRIC = {
    "3 – Transformative": ["innovative", "transformative", "impactful", "change", "improve"],
    "2 – Strategic": ["strategic", "planned", "goal", "objective", "aligned"],
    "1 – Compliant": ["followed instructions", "adhered", "standard", "compliant"],
    "0 – Counterproductive": ["ignored", "failed", "mistake", "problem", "conflict"]
}

SECTION_RUBRICS = {sec: DEFAULT_RUBRIC for sec in SECTION_MAP.values()}

# --------------------------
# 3️⃣ NLTK SETUP
# --------------------------
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# --------------------------
# 4️⃣ SEMANTIC THEME SETUP
# --------------------------
device = torch.device("cpu")  # force CPU
model = SentenceTransformer('all-MiniLM-L6-v2')  # device handled later

THEMES = {
    "Strategic": "strategic planning, decision making, stakeholder engagement",
    "Risk": "risk assessment, mitigation, safety, operational risk",
    "Communication": "messaging, transparency, framing, influence",
    "Evidence": "data analysis, research, metrics, evaluation",
    "Collaboration": "teamwork, partnership, coordination, joint action"
}

theme_texts = list(THEMES.values())
theme_embeddings = model.encode(theme_texts, convert_to_tensor=True).to(device)
theme_keys = list(THEMES.keys())

# --------------------------
# 5️⃣ FUNCTIONS
# --------------------------
@st.cache_data(ttl=300)
def fetch_kobo_data():
    """Fetch submissions from KoboToolbox and return as a DataFrame."""
    try:
        response = requests.get(KOBO_API_URL, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()
    except ValueError:
        st.error(f"Failed to parse JSON. Response text (truncated):\n{response.text[:1000]}")
        return pd.DataFrame()

    results = data if isinstance(data, list) else data.get("results", [])
    if not results:
        st.warning("No submissions found yet.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    st.write("Preview of fetched data:")
    st.dataframe(df.head())
    return df

def flatten_kobo_responses(df):
    rows = []
    for idx, row in df.iterrows():
        respondent_id = row.get("_id", f"resp_{idx}")
        for col in df.columns:
            if col.startswith("case"):
                answer = row[col]
                if pd.notna(answer) and isinstance(answer, str) and answer.strip() != "":
                    rows.append({
                        "Respondent_ID": respondent_id,
                        "Question_ID": col,
                        "Answer": answer
                    })
    return pd.DataFrame(rows)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def score_answer(answer):
    if not isinstance(answer, str) or not answer:
        return "2 – Strategic"
    answer_lower = answer.lower()
    for score, keywords in DEFAULT_RUBRIC.items():
        if any(k in answer_lower for k in keywords):
            return score
    return "2 – Strategic"

def extract_themes_with_weights(answer, top_n=3):
    if not isinstance(answer, str) or not answer.strip():
        return ""
    answer_embedding = model.encode(answer, convert_to_tensor=True).to(device)
    similarities = util.cos_sim(answer_embedding, theme_embeddings)[0]
    top_similarities, top_indices = similarities.topk(k=top_n)
    top_themes_with_weights = [
        f"{theme_keys[idx]} ({top_similarities[i].item():.2f})" for i, idx in enumerate(top_indices)
    ]
    return ", ".join(top_themes_with_weights)

def push_to_powerbi(df):
    """Push DataFrame rows to Power BI push dataset"""
    # Truncate string fields to MAX_LEN
    for col in ["Themes", "Score", "Section", "Question_ID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str[:MAX_LEN]

    data_json = df.to_dict(orient="records")
    try:
        response = requests.post(POWERBI_PUSH_URL, json=data_json)
        if response.status_code in [200, 202]:
            st.success("✅ Data successfully pushed to Power BI!")
        else:
            st.error(f"Failed to push data to Power BI: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error pushing to Power BI: {e}")

# --------------------------
# 6️⃣ STREAMLIT APP
# --------------------------
st.title("📊 Kobo Qualitative Analysis Dashboard with Power BI Push")

df = fetch_kobo_data()
if not df.empty:
    st.subheader("Raw Responses")
    st.dataframe(df.head())

    flat_df = flatten_kobo_responses(df)

    st.subheader("Scoring & Theme Extraction")
    scored_list = []

    for idx, row in flat_df.iterrows():
        qid = row["Question_ID"]
        section_prefix = "_".join(qid.split("_")[:2]) + "_group"
        section_name = SECTION_MAP.get(section_prefix, section_prefix)

        answer_text = row.get("Answer", "")
        score = score_answer(answer_text)
        themes = extract_themes_with_weights(answer_text)

        scored_row = {
            "Respondent_ID": row.get("Respondent_ID", f"resp_{idx}"),
            "Section": section_name if section_name else "Unknown Section",
            "Question_ID": qid,
            "Score": score,
            "Themes": themes
        }
        scored_list.append(scored_row)
        time.sleep(0.01)  # optional small delay

    scored_df = pd.DataFrame(scored_list)

    st.subheader("✅ Scored & Themed Responses")
    st.dataframe(scored_df)

    # Push to Power BI
    push_to_powerbi(scored_df)

    # Section summary
    if not scored_df.empty and "Section" in scored_df.columns and "Score" in scored_df.columns:
        section_summary = scored_df.groupby("Section")["Score"].value_counts().unstack(fill_value=0)
        st.subheader("Section Summary")
        st.dataframe(section_summary)
