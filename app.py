# main.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Thematic Analytics", layout="wide")

# ---- PAGES ----
import advisory
import thoughtleadership
import growthmindset
import networking
import influencingrelationship
import login  # your existing login module

# ---- make sure these keys exist ----
st.session_state.setdefault("user_email", None)
st.session_state.setdefault("show_cover", True)
st.session_state.setdefault("auth_mode", "login")
st.session_state.setdefault("is_reset_link", False)

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

# --- safe, centralized logout ---
def do_logout():
    try:
        login.sign_out()  # uses your login.py sign_out()
    finally:
        for k in ("user_email", "show_cover", "auth_mode", "is_reset_link"):
            if k in st.session_state:
                del st.session_state[k]
        # reset defaults
        st.session_state["show_cover"] = True
        st.session_state["auth_mode"] = "login"
        st.session_state["is_reset_link"] = False
        st.rerun()

def main() -> None:
    # 1) Require login first
    if not st.session_state.get("user_email"):
        login.main()   # renders your sign up / log in / reset UI
        return

    # 2) After login, show cover once (as you had)
    if st.session_state.get("show_cover", True):
        show_cover()
        return

    # 3) App navigation
    with st.sidebar:
        signed_in_as = st.session_state.get("user_email") or "Unknown"
        st.caption(f"Signed in as **{signed_in_as}**")
        selected = st.selectbox(
            "Navigation",
            options=[
                "Advisory",
                "Thought Leadership",
                "Growth Mindset",
                "Networking",
                "Influencing Relationship",
            ],
            index=0,
        )
        if st.button("Log out", use_container_width=True):
            do_logout()

    # 4) Route to pages
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
