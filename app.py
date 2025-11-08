 
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
# Full-bleed, true viewport cover (no scroll, no padding)
st.markdown("""
<style>
/* kill default paddings/margins so our cover can be full-bleed */
html, body, [data-testid="stAppViewContainer"] {
  height: 100vh !important;
  width: 100vw !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: hidden !important; /* cover is a single screen */
}
.block-container, section.main, [data-testid="stSidebar"] {
  padding: 0 !important;
  margin: 0 !important;
}
header, [data-testid="stHeader"], [data-testid="stToolbar"], footer {
  display: none !important;
}
</style>
""", unsafe_allow_html=True)


COVER_HTML = """
<div style="
  --care-orange:#EB7100;
  --care-deep:#090015;
  --text:#FFFFFF;
  --muted:#FFE7D1;
  --card:#090015;
  --card-text:#FFFFFF;
  --ring:rgba(9,0,21,0.35);
  font-family:'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;

  /* ORANGE PRIMARY BACKGROUND that fills the viewport */
  background:
    radial-gradient(1200px 700px at 10% -10%, rgba(255,255,255,0.12), transparent 60%),
    radial-gradient(900px 500px at 95% 20%, rgba(255,255,255,0.08), transparent 55%),
    linear-gradient(180deg, #F17F1C 0%, var(--care-orange) 55%, #D86400 100%);
  color: var(--text);

  /* make the section itself 100% of the viewport */
  height: 100vh; width: 100vw;
  display:flex; flex-direction:column; justify-content:space-between;
  padding: 2.25rem 2rem; box-sizing: border-box;
">
  <header style="display:flex; align-items:center; justify-content:center; gap:.75rem;">
    <div aria-hidden="true" style="width:38px;height:38px;border-radius:10px;background:#FFFFFF; box-shadow:0 8px 24px rgba(0,0,0,.18)"></div>
    <h1 style="font-size:2.2rem; font-weight:900; margin:0; letter-spacing:.2px;">Thematic Analytics</h1>
  </header>

  <main style="display:flex; flex-direction:column; align-items:center; text-align:center; margin:.5rem auto 0; max-width:1000px;">
    <h2 style="font-size:2.6rem; line-height:1.15; font-weight:900; margin:.25rem 0 1rem;">
      Qualitative Scoring for CARE Program Insights
    </h2>
    <p style="max-width:760px; font-size:1.1rem; line-height:1.6; color: var(--muted); margin:0 0 1.6rem;">
      Collect field feedback, score themes consistently, and turn narratives into evidence—so decisions move faster and impact grows.
    </p>

    <div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:center;">
      <div role="group" aria-label="Collect Feedback" style="
        background:var(--card); color:var(--card-text); border-radius:14px; width:260px; padding:1.25rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0,0,0,.22); border-top:5px solid #FFFFFF;
      ">
        <div style="font-size:1.9rem" aria-hidden="true">📝</div>
        <h3 style="margin:.35rem 0 .4rem; font-size:1.1rem; font-weight:800;">Collect Feedback</h3>
        <p style="margin:0; color:#E6E6EE;">Import transcripts, notes, and voice-to-text from the field.</p>
      </div>

      <div role="group" aria-label="Score Themes" style="
        background:var(--card); color:var(--card-text); border-radius:14px; width:260px; padding:1.25rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0,0,0,.22); border-top:5px solid #FFFFFF;
      ">
        <div style="font-size:1.9rem" aria-hidden="true">🎯</div>
        <h3 style="margin:.35rem 0 .4rem; font-size:1.1rem; font-weight:800;">Score Themes</h3>
        <p style="margin:0; color:#E6E6EE;">Apply standardized rubrics to sentiment & evidence strength.</p>
      </div>

      <div role="group" aria-label="Share Insights" style="
        background:var(--card); color:var(--card-text); border-radius:14px; width:260px; padding:1.25rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0,0,0,.22); border-top:5px solid #FFFFFF;
      ">
        <div style="font-size:1.9rem" aria-hidden="true">📊</div>
        <h3 style="margin:.35rem 0 .4rem; font-size:1.1rem; font-weight:800;">Share Insights</h3>
        <p style="margin:0; color:#E6E6EE;">Publish clear dashboards for program & MEAL teams.</p>
      </div>
    </div>

    <div style="margin-top:1.6rem; display:flex; gap:.75rem; justify-content:center; flex-wrap:wrap;">
      <span style="
        display:inline-block; font-weight:800; letter-spacing:.3px;
        background: var(--care-deep); color:#fff; padding:.85rem 1.2rem; border-radius:12px;
        box-shadow: 0 10px 24px rgba(0,0,0,.25); border: 2px solid rgba(255,255,255,.0);
      ">Get Started</span>

      <span style="
        display:inline-block; font-weight:800; letter-spacing:.3px;
        background: transparent; color:#fff; padding:.85rem 1.2rem; border-radius:12px;
        border:2px solid rgba(255,255,255,.9);
      ">Learn more</span>
    </div>
  </main>

  <footer style="font-size:.85rem; color:#FFE7D1; margin-top:.5rem; text-align:center;">
    © 2025 CARE Thematic Analytics • High contrast • Keyboard accessible
  </footer>

  <style>
    a:focus-visible, button:focus-visible, span:focus-visible {
      outline: 3px solid var(--ring); outline-offset: 3px; border-radius: 10px;
      box-shadow: 0 0 0 2px #FFFFFF;
    }
    @media (max-width: 680px){
      h2{ font-size: 1.9rem !important; }
    }
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





