# main.py
import streamlit as st
import streamlit.components.v1 as components

# === PAGES (make sure these files exist under ./pages/)
from pages import advisory_page
from pages import thought_leadership_page
from pages import growth_mindset_skills_page
from pages import networking_and_advocacy_page
from pages import influencing_relationship_page  # <-- not "...pageso"

st.set_page_config(page_title="DATA LENS", page_icon="ASIGMA.png", layout="wide")

# ----- optional cover -----
COVER_HTML = """
<div style="font-family: Inter, sans-serif; background:#f6f7f8; height: 100vh; display:flex; flex-direction:column; justify-content:space-between; text-align:center; color:#101922; padding:2rem 3rem;">
  <div>
    <h1 style="font-size:2.8rem; font-weight:900; margin-bottom:.75rem;">Streamline Your Data Workflow</h1>
    <p style="font-size:1.1rem; color:#4b5563; max-width:600px; margin:0 auto 1.8rem; line-height:1.5;">
      Powerful tools for data cleaning, database management, and insightful analytics.
    </p>
  </div>
  <footer style="font-size:.85rem; color:#777; margin-top:1rem;">© 2025 DataLens. All rights reserved.</footer>
</div>
"""

def show_cover():
    components.html(COVER_HTML, height=550, scrolling=False)
    st.write("")
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state.show_cover = False
        st.rerun()

def main():
    st.session_state.setdefault("show_cover", True)
    st.session_state.setdefault("user_email", True)  # set True if you skip auth; integrate your auth if needed

    if st.session_state["show_cover"]:
        show_cover()
        return

    with st.sidebar:
        selected = st.selectbox(
            "Navigation",
            [
                "Advisory",
                "Thought Leadership",
                "Growth Mindset Skills",
                "Networking and Advocacy",
                "Influencing Relationship",
            ],
        )

    if selected == "Advisory":
        advisory_page.main()
    elif selected == "Thought Leadership":
        thought_leadership_page.main()
    elif selected == "Growth Mindset Skills":
        growth_mindset_skills_page.main()
    elif selected == "Networking and Advocacy":
        networking_and_advocacy_page.main()
    elif selected == "Influencing Relationship":
        influencing_relationship_page.main()

if __name__ == "__main__":
    main()
