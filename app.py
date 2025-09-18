import streamlit as st
import requests
import pandas as pd
import time
# --------------------------
# 1️⃣ CONFIG
# --------------------------
KOBO_FORM_ID = "atdspJQv7RBwjkmaVFRS43"
KOBO_API_URL = f"https://kf.kobotoolbox.org/assets/atdspJQv7RBwjkmaVFRS43/submissions/?format=json"

KOBO_TOKEN = st.secrets["KOBO_TOKEN"]
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]  # Hugging Face Inference API token
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# 2️⃣ RUBRICS FOR FIVE SECTIONS
# --------------------------
SECTIONS = {
    "Influencing": """
3 – Transformative: co-creates direction; transparent ownership; empathic dialogue; deep field-first orientation.
2 – Strategic: risk-aware, structured, protocol-aligned, but less co-created or field-led.
1 – Compliant: procedural fixes, workarounds, or passivity without addressing root causes.
0 – Counterproductive: abdicates responsibility, prioritizes image/control, or escalates without dialogue.
""",
    "Advisory Skills": """
3 – Transformative: co-creates direction; transparent ownership; empathic dialogue; deep field-first orientation; mutual partnership reset.
2 – Strategic: risk-aware, structured, protocol-aligned, but less co-created or field-led.
1 – Compliant: procedural fixes, workarounds, or passivity without addressing root causes.
0 – Counterproductive: abdicates responsibility, prioritizes image/control, or escalates without dialogue.
""",
    "Strategic Thought Leadership": """
3 – Transformative: sets bold, executable strategy; field-first; adaptive learning; aligned cross-functional delivery.
2 – Strategic: structured, aligned, but less locally grounded.
1 – Compliant: procedural execution without adaptation or insight.
0 – Counterproductive: drifts from vision or erodes trust.
""",
    "Networking & Advocacy": """
3 – Transformative: builds inclusive coalitions; advances local leadership; risk-aware advocacy; ethical evidence use.
2 – Strategic: manages coalitions; risk-aware, but limited inclusion or evidence.
1 – Compliant: procedural, limited advocacy; reactive to risk.
0 – Counterproductive: undermines local voice; escalates risk; ignores evidence.
""",
    "Growth Mindset": """
3 – Transformative: rapid learning; iterative adaptation; field-led innovation; ethical and inclusive experimentation.
2 – Strategic: risk-aware learning; structured adaptation; less co-created or field-led.
1 – Compliant: procedural adaptation without insight or innovation.
0 – Counterproductive: ignores learning opportunities; rigid or unsafe adaptation.
"""
}
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
def hf_score_answer(answer, rubric, section):
    """Send answer to Hugging Face API to get score + themes."""
    if not isinstance(answer, str) or answer.strip() == "":
        return pd.Series(["No response", "", section])

    prompt = f"""
Candidate answer: {answer}

Rubric: {rubric}

Task: Summarize key behaviors, extract themes, suggest a score (0-3), and give a one-sentence justification.
"""
    api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    payload = {"inputs": prompt}
    try:
        response = requests.post(api_url, headers=HF_HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        result_text = response.json()[0]['generated_text']
    except Exception as e:
        return pd.Series(["Error", str(e), section])

    # Try to parse score + themes
    score = "?"
    themes = ""
    for s in ["0", "1", "2", "3"]:
        if f"{s}" in result_text:
            score = s
            break
    # Extract themes heuristically (after "Themes:" or "themes:")
    if "Themes:" in result_text:
        themes = result_text.split("Themes:")[1].split("\n")[0].strip()
    elif "themes:" in result_text:
        themes = result_text.split("themes:")[1].split("\n")[0].strip()

    return pd.Series([score, themes, section])

# --------------------------
# 3️⃣ STREAMLIT APP
# --------------------------
st.title("📊 Kobo AI Dashboard with Automated Scoring & Themes")
st.markdown("""
Fetches responses from **KoboToolbox**, scores them per section using AI, extracts themes, and displays a clean table ready for Power BI.
""")

df = fetch_kobo_data()

if not df.empty:
    st.subheader("Raw Kobo Responses")
    st.dataframe(df.head())

    # Prepare dataframe for scoring
    scores_list = []
    for idx, row in df.iterrows():
        for section, rubric in SECTIONS.items():
            answer_text = row.get(section, "")
            score, themes, sec = hf_score_answer(answer_text, rubric, section)
            scores_list.append({
                "Respondent_ID": row.get("_id", idx),
                "Section": sec,
                "Answer": answer_text,
                "Score": score,
                "Themes": themes
            })
            time.sleep(0.5)  # avoid HF free tier rate limits

    scored_df = pd.DataFrame(scores_list)

    st.subheader("✅ Scored & Themed Responses")
    st.dataframe(scored_df)

    st.markdown("This table can be pushed to Power BI for live dashbord")
