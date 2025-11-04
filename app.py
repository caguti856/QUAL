# main.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Thematic Analytics",  layout="wide")

# === PAGES (make sure advisory.py exists in the same folder or is importable)
import advisory  # must define advisory.main()
import thought_leadership
# ----- optional cover -----
COVER_HTML = """
<div style="font-family: Inter, sans-serif; background:#f6f7f8; height: 100vh; display:flex; flex-direction:column; justify-content:space-between; text-align:center; color:#101922; padding:2rem 3rem;">
  <div>
    <h1 style="font-size:2.8rem; font-weight:900; margin-bottom:.75rem;">Streamline Your Data Workflow</h1>
    <p style="font-size:1.1rem; color:#4b5563; max-width:600px; margin:0 auto 1.8rem; line-height:1.5;">
      Powerful tools for data cleaning, database management, and insightful analytics.
    </p>
  </div>
  <footer style="font-size:.85rem; color:#777; margin-top:1rem;">© Thematic Analytics CARE UGANDA. All rights reserved.</footer>
</div>
"""

def show_cover() -> None:
    components.html(COVER_HTML, height=550, scrolling=False)
    st.write("")
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state["show_cover"] = False
        st.rerun()

def main() -> None:
    st.session_state.setdefault("show_cover", True)
    
    if st.session_state["show_cover"]:
        show_cover()
        return

    with st.sidebar:
        selected = st.selectbox(
            "Navigation",
            options=["Advisory", "Thought Leadership"],  # <-- SINGLE list
            index=0,
        )

    # ✅ correct branching + call
    if selected == "Advisory":
        advisory.main()
    elif selected == "Thought Leadership":
        thought_leadership.main()
if __name__ == "__main__":
    main()
