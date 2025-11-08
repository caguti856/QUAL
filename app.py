 
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Thematic Analytics",  layout="wide")

# === PAGES (make sure advisory.py exists in the same folder or is importable)
import advisory  # must define advisory.main()
import thoughtleadership
import growthmindset
import networking
import influencingrelationship
import login
# ----- optional cover -----

COVER_HTML = """
<div style="
  --care-orange:#EB7100;
  --care-deep:#090015;
  --text:#FFFFFF;
  --muted:#D7D7DB;
  --card:#FFFFFF;
  --card-text:#1E1E2A;
  --ring:rgba(235,113,0,0.35);
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  background:
    radial-gradient(1200px 700px at 10% -10%, rgba(235,113,0,0.18), transparent 60%),
    radial-gradient(900px 500px at 95% 20%, rgba(235,113,0,0.12), transparent 55%),
    linear-gradient(180deg, var(--care-deep) 0%, #0E0A1E 100%);
  color: var(--text);
  height: 100vh; display: flex; flex-direction: column; justify-content: space-between;
  padding: 2.25rem 2rem;
">
  <header style="display:flex; align-items:center; justify-content:center; gap:.75rem;">
    <div aria-hidden="true" style="width:38px;height:38px;border-radius:10px;background:var(--care-orange); box-shadow:0 8px 24px rgba(235,113,0,.35)"></div>
    <h1 style="font-size:2.2rem; font-weight:900; margin:0; letter-spacing:.2px;">Thematic Analytics</h1>
  </header>

  <main style="display:flex; flex-direction:column; align-items:center; text-align:center; margin: .5rem auto 0; max-width:1000px;">
    <h2 style="font-size:2.6rem; line-height:1.15; font-weight:900; margin:.25rem 0 1rem;">
      Qualitative Scoring for CARE Program Insights
    </h2>
    <p style="max-width:760px; font-size:1.1rem; line-height:1.6; color: var(--muted); margin:0 0 1.6rem;">
      Collect field feedback, score themes consistently, and turn narratives into evidence—so decisions move faster and impact grows.
    </p>

    <div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:center;">
      <div role="group" aria-label="Collect Feedback" style="
        background:var(--card); color:var(--card-text); border-radius:14px; width:260px; padding:1.25rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0,0,0,.18); border-top:5px solid var(--care-orange);
      ">
        <div style="font-size:1.9rem" aria-hidden="true">📝</div>
        <h3 style="margin:.35rem 0 .4rem; font-size:1.1rem; font-weight:800;">Collect Feedback</h3>
        <p style="margin:0; color:#3B3B48;">Import transcripts, notes, and voice-to-text from the field.</p>
      </div>

      <div role="group" aria-label="Score Themes" style="
        background:var(--card); color:var(--card-text); border-radius:14px; width:260px; padding:1.25rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0,0,0,.18); border-top:5px solid var(--care-orange);
      ">
        <div style="font-size:1.9rem" aria-hidden="true">🎯</div>
        <h3 style="margin:.35rem 0 .4rem; font-size:1.1rem; font-weight:800;">Score Themes</h3>
        <p style="margin:0; color:#3B3B48;">Apply standardized rubrics to sentiment & evidence strength.</p>
      </div>

      <div role="group" aria-label="Share Insights" style="
        background:var(--card); color:var(--card-text); border-radius:14px; width:260px; padding:1.25rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0,0,0,.18); border-top:5px solid var(--care-orange);
      ">
        <div style="font-size:1.9rem" aria-hidden="true">📊</div>
        <h3 style="margin:.35rem 0 .4rem; font-size:1.1rem; font-weight:800;">Share Insights</h3>
        <p style="margin:0; color:#3B3B48;">Publish clear dashboards for program & MEAL teams.</p>
      </div>
    </div>
  </main>

  <footer style="font-size:.85rem; color:#CFCFD6; margin-top: .5rem; text-align:center;">
    © 2025 CARE Thematic Analytics • High contrast • Keyboard accessible
  </footer>

  <style>
    /* Focus-visible outlines for a11y */
    a:focus-visible, button:focus-visible {
      outline: 3px solid var(--ring); outline-offset: 3px; border-radius: 10px;
    }
    /* Larger text on small screens */
    @media (max-width: 680px){
      h2{ font-size: 1.9rem !important; }
    }
    /* Respect reduced motion */
    @media (prefers-reduced-motion: reduce){
      * { animation: none !important; transition: none !important; }
    }
  </style>
</div>
"""




# --- GLOBAL SESSION SETUP ---
# ✅ Only defined ONCE, here in main.py
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "show_cover" not in st.session_state:
    st.session_state["show_cover"] = True
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"
# --- Cover page content (HTML + CSS) ---


# --- Show cover page with native Streamlit button ---
def show_cover_page():
    # Render HTML
    components.html(COVER_HTML, height=650, scrolling=False)
    st.write("")  # Add small padding
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        clicked = st.button("🚀 Get Started", use_container_width=True)
    if clicked:
        st.session_state.show_cover = False
        st.session_state.auth_mode = "login"
        st.rerun()
  

def main():
    if 'show_cover' not in st.session_state:
        st.session_state['show_cover'] = True
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = None
    if 'auth_mode' not in st.session_state:
        st.session_state['auth_mode'] = "login"

    # Routing logic
    if st.session_state['show_cover']:
        show_cover_page()
    else:
        if not st.session_state['user_email']:
            login.show_auth_page()
        else:
            # Your sidebar navigation as before
          with st.sidebar:
                selected = st.selectbox(
                "Navigation",
                ["Advisory", "Thought Leadership","Growth Mindset","Networking","Influencing Relationship","Logout"]
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
          elif selected == "Logout":
             st.session_state.user_email = None
             st.session_state['show_cover'] = True


if __name__ == "__main__":
    main()





