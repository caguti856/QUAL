import streamlit as st
import requests
import pandas as pd
import os
KOBO_TOKEN = os.getenv("KOBO_TOKEN")
# --------------------------
# 1. CONFIG
# --------------------------
# Replace with your KoboToolbox API token

KOBO_FORM_ID = "atdspJQv7RBwjkmaVFRS43"  # e.g. "a1b2c3d4..." from Kobo form URL

KOBO_API_URL = f"https://kf.kobotoolbox.org/api/v2/assets/{atdspJQv7RBwjkmaVFRS43}/data/"

HEADERS = {
    "Authorization": f"Token {KOBO_TOKEN}"
}

# --------------------------
# 2. FUNCTIONS
# --------------------------
@st.cache_data(ttl=300)  # cache for 5 min
def fetch_kobo_data():
    """Fetch responses from Kobo API"""
    response = requests.get(KOBO_API_URL, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        return pd.DataFrame(results)
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return pd.DataFrame()

# --------------------------
# 3. STREAMLIT APP
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
