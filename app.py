# main.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Thematic Analytics", layout="wide")

# pages
import advisory
import thoughtleadership
import growthmindset
import networking
import influencingrelationship

# auth
import auth  # <-- the new file above

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

def show_cover():
    components.html(COVER_HTML, height=550, scrolling=False)
    st.write("")
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state["show_cover"] = False
        st.rerun()

def main():
    st.session_state.setdefault("show_cover", True)

    # 🔒 Require login before anything else (cover is shown only after login)
    if not auth.require_auth():
        return

    # Top bar: who & logout
    with st.sidebar:
        st.caption(f"Signed in as **{st.session_state.get('user_email','')}**")
        if st.button("Log out"):
            auth.sign_out()
            st.rerun()

    # Optional: cover page after login
    if st.session_state["show_cover"]:
        show_cover()
        return

    # App navigation (visible only when logged in)
    with st.sidebar:
        selected = st.selectbox(
            "Navigation",
            options=[
                "Advisory", "Thought Leadership", "Growth Mindset",
                "Networking", "Influencing Relationship"
            ],
            index=0,
        )

    if selected == "Advisory":
        advisory.main()
    elif selected == "Thought Leadership":
        thoughtleadership.main()
    elif selected == "Growth Mindset":
        growthmindset.main()
    elif selected == "Networking":
        networking.main()
    elif selected == "Influencing Relationship":
        influencingrelationship.main()

if __name__ == "__main__":
    main()
