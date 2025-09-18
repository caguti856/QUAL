import streamlit as st
import requests
import pandas as pd
import time
from transformers import pipeline
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
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

model_pipeline = load_model()

def hf_score_batch(answers, rubrics, sections):
    prompts = []
    for answer, rubric in zip(answers, rubrics):
        prompts.append(f"""
Candidate answer: {answer if answer.strip() else '[No response]'}
Rubric: {rubric}
Task: Summarize key behaviors, extract themes, suggest a score (0-3), one-sentence justification.
""")
    
    results = model_pipeline(prompts, max_length=200)
    
    scored = []
    for res, sec in zip(results, sections):
        text = res['generated_text']
        score = next((s for s in ["0","1","2","3"] if s in text), "?")
        themes = ""
        if "Themes:" in text:
            themes = text.split("Themes:")[1].split("\n")[0].strip()
        elif "themes:" in text:
            themes = text.split("themes:")[1].split("\n")[0].strip()
        scored.append((score, themes, sec))
    return scored

# --------------------------
# 5️⃣ STREAMLIT APP
# --------------------------
st.title("📊 Kobo AI Dashboard (Batch Transformers)")

df = fetch_kobo_data()
if not df.empty:
    st.subheader("Raw Responses")
    st.dataframe(df.head())

    flat_df = flatten_kobo_responses(df)
    
    st.subheader("Scoring with AI (Batch)...")
    batch_size = 20  # adjust batch size for speed/memory
    scored_list = []

    for i in range(0, len(flat_df), batch_size):
        batch = flat_df.iloc[i:i+batch_size]

        sections, rubrics = [], []
        for qid in batch["Question_ID"]:
            prefix = "_".join(qid.split("_")[:2]) + "_group"
            sections.append(SECTION_MAP.get(prefix, prefix))
            rubrics.append(SECTION_RUBRICS.get(SECTION_MAP.get(prefix, prefix), DEFAULT_RUBRIC))
        
        scored_batch = hf_score_batch(batch["Answer"].tolist(), rubrics, sections)
        for (score, themes, sec), (_, row) in zip(scored_batch, batch.iterrows()):
            scored_list.append({
                "Respondent_ID": row["Respondent_ID"],
                "Section": sec,
                "Question_ID": row["Question_ID"],
                "Answer": row["Answer"],
                "Score": score,
                "Themes": themes
            })
    
    scored_df = pd.DataFrame(scored_list)
    st.subheader("✅ AI Scored & Themed Responses")
    st.dataframe(scored_df)
