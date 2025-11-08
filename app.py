 
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
html, body, [data-testid="stAppViewContainer"]{
  height:100vh !important; width:100vw !important;
  padding:0 !important; margin:0 !important; overflow:hidden !important;
}
.block-container, section.main{ padding:0 !important; margin:0 !important; }
header, [data-testid="stHeader"], [data-testid="stToolbar"], footer{ display:none !important; }
</style>
""", unsafe_allow_html=True)



COVER_HTML = """
<div style="
  --care-orange:#EB7100;       /* Primary brand */
  --care-orange-2:#ff8a1f;     /* Lighter stop for gradient */
  --care-deep:#090015;         /* Accessible deep accent */
  --text:#fff;
  --muted:#FFEAD6;             /* Warm, readable supporting text */
  --card:#0B0716;
  --ring:rgba(9,0,21,.35);
  --radius:18px;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;

  /* Full-bleed, responsive hero */
  height:100vh; width:100vw; box-sizing:border-box;
  display:flex; flex-direction:column; justify-content:space-between;
  padding: clamp(20px, 3vw, 36px);
  color:var(--text);

  /* Warm orange gradient with subtle glow */
  background:
    radial-gradient(1200px 700px at 12% -10%, rgba(255,255,255,.14), transparent 60%),
    radial-gradient(900px 600px at 92% 15%, rgba(255,255,255,.12), transparent 55%),
    linear-gradient(180deg, var(--care-orange-2) 0%, var(--care-orange) 60%, #D35F00 100%);
">

  <!-- Top brand bar -->
  <header style="display:flex; align-items:center; justify-content:center; gap:.85rem;">
    <div aria-hidden="true" style="
      width:40px; height:40px; border-radius:12px; background:#fff;
      box-shadow:0 10px 26px rgba(0,0,0,.22);
    "></div>
    <h1 style="margin:0; font-size:clamp(1.4rem, 2.2vw, 2.2rem); font-weight:900; letter-spacing:.2px;">
      Thematic Analytics
    </h1>
  </header>

  <!-- Main content -->
  <main style="max-width:1120px; margin: 0 auto; text-align:center;">
    <h2 style="
      margin:.25rem auto 1rem;
      font-weight:900; line-height:1.08;
      font-size:clamp(1.8rem, 5vw, 3.2rem);
      text-shadow:0 2px 0 rgba(0,0,0,.12);
    ">
      Qualitative Scoring for CARE Program Insights
    </h2>

    <p style="
      margin:0 auto 1.8rem;
      max-width:820px; color:var(--muted);
      font-size:clamp(1rem, 1.4vw, 1.15rem); line-height:1.65;
    ">
      Collect field feedback, score themes consistently, and turn narratives into evidence—so decisions move faster and impact grows.
    </p>

    <!-- Feature cards -->
    <div style="
      display:grid; gap:clamp(14px, 2vw, 24px);
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      align-items:stretch; margin: 0 auto;
      max-width: 1100px;
    ">
      <!-- Card -->
      <div style="
        background:var(--card); border-radius:var(--radius);
        padding: clamp(16px, 2vw, 24px);
        border:2px solid rgba(255,255,255,.12);
        box-shadow: 0 14px 38px rgba(0,0,0,.25);
      ">
        <div style="font-size:1.9rem; margin-bottom:.4rem">📝</div>
        <h3 style="margin:.2rem 0 .35rem; font-size:1.2rem; font-weight:800;">Collect Feedback</h3>
        <p style="margin:0; color:#E7E7F2">Import transcripts, notes, and voice-to-text from the field.</p>
      </div>

      <div style="
        background:var(--card); border-radius:var(--radius);
        padding: clamp(16px, 2vw, 24px);
        border:2px solid rgba(255,255,255,.12);
        box-shadow: 0 14px 38px rgba(0,0,0,.25);
      ">
        <div style="font-size:1.9rem; margin-bottom:.4rem">🎯</div>
        <h3 style="margin:.2rem 0 .35rem; font-size:1.2rem; font-weight:800;">Score Themes</h3>
        <p style="margin:0; color:#E7E7F2">Use consistent rubrics for sentiment & evidence strength.</p>
      </div>

      <div style="
        background:var(--card); border-radius:var(--radius);
        padding: clamp(16px, 2vw, 24px);
        border:2px solid rgba(255,255,255,.12);
        box-shadow: 0 14px 38px rgba(0,0,0,.25);
      ">
        <div style="font-size:1.9rem; margin-bottom:.4rem">📊</div>
        <h3 style="margin:.2rem 0 .35rem; font-size:1.2rem; font-weight:800;">Share Insights</h3>
        <p style="margin:0; color:#E7E7F2">Publish clear dashboards for program & MEAL teams.</p>
      </div>
    </div>

    <!-- CTAs -->
  </main>

  <footer style="
    color:#FFEAD6; text-align:center; font-size:.9rem; margin-top:.6rem;
  ">
    © 2025 CARE Thematic Analytics • WCAG AA contrast • Keyboard accessible
  </footer>

  <style>
    /* Responsive grid on small screens */
    @media (max-width: 980px){
      main > div { grid-template-columns: 1fr; }
    }
    /* Focus rings for keyboard users */
    a:focus-visible {
      outline: 3px solid var(--ring); outline-offset: 3px; border-radius: 12px;
      box-shadow: 0 0 0 2px #fff;
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





