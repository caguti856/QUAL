import pandas as pd
import streamlit as st
import requests

KOBO_API_URL = "https://kf.kobotoolbox.org/api/v2/assets/atdspJQv7RBwjkmaVFRS43/data/"
HEADERS = {"Authorization": f"Token {st.secrets['KOBO_TOKEN']}"}

@st.cache_data(ttl=300)
def fetch_kobo_data():
    response = requests.get(KOBO_API_URL, headers=HEADERS)
    st.write(f"Status code: {response.status_code}")
    try:
        data = response.json()
        st.write("Raw JSON response:", data)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return pd.DataFrame()

    # Check where the actual submissions are
    results = data.get("results") or []
    if not results:
        st.warning("No submissions found or JSON structure unexpected.")
    return pd.DataFrame(results)
