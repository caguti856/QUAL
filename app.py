# main.py
import streamlit as st
import streamlit.components.v1 as components

import login
import advisory
import thoughtleadership
import growthmindset
import networking
import influencingrelationship

st.set_page_config(page_title="Thematic Analytics", layout="wide")
st.session_state.setdefault("user_email", None)
st.session_state.setdefault("show_cover", True)

def show_cover():
    components.html("...your cover html...", height=550, scrolling=False)
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state["show_cover"] = False
        st.rerun()

def main():
    # 1) Gate by login
    if not st.session_state.get("user_email"):
        login.main()
        return

    # 2) Optional cover once
    if st.session_state.get("show_cover", True):
        show_cover()
        return

    # 3) Sidebar + logout
    with st.sidebar:
        st.caption(f"Signed in as **{st.session_state['user_email']}**")
        selected = st.selectbox(
            "Navigation",
            ["Advisory", "Thought Leadership", "Growth Mindset", "Networking", "Influencing Relationship"],
            index=0
        )
        if st.button("Log out"):
            login.sign_out()
            st.rerun()

    # 4) Router
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
