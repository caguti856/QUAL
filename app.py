 
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



# ----- optional cover (CARE colors, full-bleed, no external deps) -----
COVER_HTML = """
<div style="
  --care-orange:#EB7100;
  --care-deep:#090015;
  --text:#FFFFFF;
  --muted:rgba(255,255,255,.75);
  --card:#0B0712;
  --ring:rgba(235,113,0,.45);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  min-height: 100vh; height: 100vh; width:100%;
  display:flex; flex-direction:column;
  background:
    radial-gradient(1200px 600px at -10% -10%, rgba(235,113,0,.18), transparent 60%),
    radial-gradient(900px 500px at 110% 0%, rgba(235,113,0,.12), transparent 55%),
    linear-gradient(180deg, var(--care-orange) 0%, #D76400 26%, #B55500 60%, var(--care-deep) 100%);
  color: var(--text);
">

  <!-- header -->
  <header style="
    display:flex; align-items:center; justify-content:space-between;
    padding: 18px clamp(16px, 5vw, 48px); border-bottom: 1px solid rgba(255,255,255,.18);
  ">
    <div style="display:flex; gap:.75rem; align-items:center;">
      <div aria-hidden="true" style="width:36px;height:36px;border-radius:10px;background:#fff;opacity:.9;"></div>
      <h2 style="margin:0; font-weight:800; letter-spacing:-.015em;">Thematic Analytics</h2>
    </div>

    <nav style="display:flex; gap:28px;">
      <a href="#" style="color:rgba(255,255,255,.8); text-decoration:none; font-weight:600;">Features</a>
      <a href="#" style="color:rgba(255,255,255,.8); text-decoration:none; font-weight:600;">How it works</a>
      <a href="#" style="color:rgba(255,255,255,.8); text-decoration:none; font-weight:600;">Support</a>
    </nav>
  </header>

  <!-- hero -->
  <main style="flex:1; display:flex; align-items:center; justify-content:center;">
    <section style="width:min(1100px, 100%); padding: clamp(16px, 5vw, 48px); text-align:center;">
      <h1 style="
        margin:0 0 16px 0;
        font-size: clamp(36px, 6vw, 76px);
        line-height:1.05; font-weight:900; letter-spacing:-.02em; text-wrap:balance;
        text-shadow: 0 8px 28px rgba(0,0,0,.25);
      ">
        Consistent scoring.<br>Credible insights.<br>Bigger impact.
      </h1>

      <p style="
        margin:0 auto 28px auto; max-width: 820px;
        font-size: clamp(16px, 1.6vw, 22px); line-height:1.55; color: var(--muted);
      ">
        From field voices to reliable evidence—so teams act faster and smarter.
      </p>

      <!-- feature cards -->
      <div style="
        display:grid; gap:18px;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 28px auto 36px; max-width: 1100px;
      ">
        <div style="
          background: rgba(9,0,21,.75);
          border:1px solid rgba(255,255,255,.1);
          border-radius:18px; padding:22px 20px; box-shadow: 0 16px 40px rgba(0,0,0,.25);
        ">
          <div style="font-size:28px; margin-bottom:6px;">📝</div>
          <h3 style="margin:0 0 6px 0; font-size:20px; font-weight:800;">Collect Feedback</h3>
          <p style="margin:0; color:rgba(255,255,255,.8)">Import transcripts, notes, and voice-to-text from the field.</p>
        </div>

        <div style="
          background: rgba(9,0,21,.75);
          border:1px solid rgba(255,255,255,.1);
          border-radius:18px; padding:22px 20px; box-shadow: 0 16px 40px rgba(0,0,0,.25);
        ">
          <div style="font-size:28px; margin-bottom:6px;">🎯</div>
          <h3 style="margin:0 0 6px 0; font-size:20px; font-weight:800;">Score Themes</h3>
          <p style="margin:0; color:rgba(255,255,255,.8)">Apply standardized rubrics to sentiment & evidence strength.</p>
        </div>

        <div style="
          background: rgba(9,0,21,.75);
          border:1px solid rgba(255,255,255,.1);
          border-radius:18px; padding:22px 20px; box-shadow: 0 16px 40px rgba(0,0,0,.25);
        ">
          <div style="font-size:28px; margin-bottom:6px;">📊</div>
          <h3 style="margin:0 0 6px 0; font-size:20px; font-weight:800;">Share Insights</h3>
          <p style="margin:0; color:rgba(255,255,255,.8)">Publish clear dashboards for program & MEAL teams.</p>
        </div>
      </div>

      <!-- buttons are just visual; actual action is your Streamlit button below -->
      <div style="display:flex; gap:14px; justify-content:center; flex-wrap:wrap;">
        <a role="button" href="#" style="
          background:#fff; color:#1a1a1a; padding:14px 22px; font-weight:800;
          border-radius:14px; text-decoration:none; box-shadow:0 18px 40px rgba(0,0,0,.30);
        ">Get Started</a>
        <a role="button" href="#" style="
          background:transparent; color:#fff; padding:14px 22px; font-weight:800;
          border-radius:14px; text-decoration:none; border:2px solid rgba(255,255,255,.6);
        ">Learn more</a>
      </div>
    </section>
  </main>

  <footer style="
    border-top:1px solid rgba(255,255,255,.18);
    padding: 18px clamp(16px, 5vw, 48px); text-align:center; color: rgba(255,255,255,.7);
  ">
    © 2025 CARE Thematic Analytics • WCAG contrast • Keyboard accessible
  </footer>

  <style>
    @media (max-width: 900px){
      .cards { grid-template-columns: 1fr !important; }
    }
    a:focus-visible, button:focus-visible {
      outline: 3px solid var(--ring); outline-offset: 3px; border-radius: 12px;
    }
    @media (prefers-reduced-motion: reduce){ * { transition:none !important; animation:none !important; } }
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





