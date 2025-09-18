import streamlit as st
import requests
import pandas as pd

# --------------------------
# 1️⃣ CONFIG
# --------------------------
KOBO_FORM_ID = "atdspJQv7RBwjkmaVFRS43"
KOBO_API_URL = f"https://kf.kobotoolbox.org/assets/atdspJQv7RBwjkmaVFRS43/submissions/?format=json"

KOBO_TOKEN = st.secrets["KOBO_TOKEN"]
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}

# --------------------------
# 2️⃣ FUNCTIONS
# --------------------------
@st.cache_data(ttl=300)
def fetch_kobo_data():
    try:
        response = requests.get(KOBO_API_URL, headers=HEADERS)
        st.write(f"Status code: {response.status_code}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from Kobo API: {e}")
        return pd.DataFrame()

    # Try to parse JSON safely
    try:
        data = response.json()
    except ValueError as e:
        st.error(f"Failed to parse JSON. Response text:\n{response.text[:1000]}")
        return pd.DataFrame()

    # Check submissions
    results = data.get("results") or []
    if not results:
        st.warning("No submissions found yet.")
        return pd.DataFrame()

    return pd.DataFrame(results)

# --------------------------
# 3️⃣ STREAMLIT APP
# --------------------------
st.title("📊 Kobo Questionnaire Dashboard")
st.markdown("""
This app fetches responses from **KoboToolbox** in real-time and displays them here.  
Next steps: add scoring logic + send results to Power BI.
""")

df = fetch_kobo_data()

if not df.empty:
    st.subheader("Raw Responses")
    st.dataframe(df)

    st.subheader("Summary")
    st.write(df.describe(include="all"))
else:
    st.warning("No data yet or error fetching responses.")

