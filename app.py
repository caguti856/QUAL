import streamlit as st
import requests
import pandas as pd

# --------------------------
# 1️⃣ CONFIG
# --------------------------
KOBO_FORM_ID = "atdspJQv7RBwjkmaVFRS43"  # Kobo form ID
KOBO_API_URL = f"https://kf.kobotoolbox.org/api/v2/assets/{KOBO_FORM_ID}/data/"

# Get your Kobo token from Streamlit secrets
KOBO_TOKEN = st.secrets["KOBO_TOKEN"]
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}

# --------------------------
# 2️⃣ FUNCTIONS
# --------------------------
@st.cache_data(ttl=300)
def fetch_kobo_data():
    """Fetch submissions from Kobo API"""
    try:
        response = requests.get(KOBO_API_URL, headers=HEADERS)
        st.write(f"Status code: {response.status_code}")
        response.raise_for_status()  # Raise error for HTTP issues
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from Kobo API: {e}")
        return pd.DataFrame()

    try:
        data = response.json()
        st.write("Raw JSON response (preview):", data if len(str(data)) < 1000 else str(data)[:1000] + "...")
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return pd.DataFrame()

    # Check where the actual submissions are
    results = data.get("results") or data.get("submissions") or []
    if not results:
        st.warning("No submissions found or JSON structure unexpected.")
        return pd.DataFrame()

    # Convert to DataFrame
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
