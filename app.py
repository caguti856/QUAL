 
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
  --muted:rgba(255,255,255,.78);
  --ring:rgba(235,113,0,.45);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  min-height:100vh; height:100vh; width:100%;
  display:flex; flex-direction:column;
  background:
    radial-gradient(1100px 600px at 20% -10%, rgba(255,255,255,.08), transparent 60%),
    radial-gradient(900px 500px at 110% 0%, rgba(255,255,255,.06), transparent 55%),
    linear-gradient(180deg, #F07F1C 0%, #EB7100 22%, #C25E06 56%, #7C3A00 82%, var(--care-deep) 100%);
  color:var(--text);
  overflow:hidden;
">

  <!-- Top bar -->
  <header style="
    display:flex; align-items:center; justify-content:space-between;
    padding:18px clamp(16px,5vw,48px); border-bottom:1px solid rgba(255,255,255,.12);
    backdrop-filter:saturate(140%) blur(0px);
  ">
    <div style="display:flex; gap:.75rem; align-items:center;">
      <div aria-hidden="true" style="width:32px;height:32px;border-radius:9px;background:#fff; opacity:.9; box-shadow:0 6px 18px rgba(0,0,0,.25)"></div>
      <h2 style="margin:0; font-weight:900; letter-spacing:-.015em;">Thematic Analytics</h2>
    </div>
    <nav style="display:flex; gap:18px; opacity:.9">
      <span style="font-size:.95rem;">CARE-ready</span>
      <span style="font-size:.95rem;">WCAG AA</span>
    </nav>
  </header>

  <!-- Hero -->
  <main style="flex:1; display:grid; place-items:center; padding: clamp(12px,4vw,32px);">
    <section style="text-align:center; max-width:1100px; padding:0 clamp(12px,4vw,40px);">
      <h1 style="
        margin:0 0 12px 0;
        font-size:clamp(38px,6.5vw,86px);
        line-height:1.02; font-weight:900; letter-spacing:-.02em; text-wrap:balance;
        text-shadow:0 10px 32px rgba(0,0,0,.28);
      ">
        Consistent scoring. <span style="opacity:.95">Credible insights.</span> Bigger impact.
      </h1>
      <p style="
        margin:8px auto 28px; max-width:800px;
        font-size:clamp(16px,2.2vw,20px); line-height:1.6; color:var(--muted)
      ">
        Turn qualitative feedback into defensible, program-ready evidence—fast.
      </p>

      <!-- CTA group -->
      <div style="display:flex; gap:14px; justify-content:center; flex-wrap:wrap;">
        <a role="button" href="?start=1" style="
          display:inline-flex; align-items:center; gap:.55rem;
          background:#fff; color:#111; font-weight:800;
          padding:14px 22px; border-radius:999px; text-decoration:none;
          box-shadow:0 10px 28px rgba(0,0,0,.28);
          transform: translateZ(0); transition: transform .15s ease, box-shadow .15s ease;
        "
           onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 16px 38px rgba(0,0,0,.32)'"
           onmouseout="this.style.transform=''; this.style.boxShadow='0 10px 28px rgba(0,0,0,.28)'">
          🚀 Get Started
        </a>
        <a role="button" href="#learn" style="
          display:inline-flex; align-items:center; gap:.55rem;
          background:transparent; color:#fff; border:2px solid rgba(255,255,255,.75);
          font-weight:800; padding:12px 20px; border-radius:999px; text-decoration:none;
        ">Learn more</a>
      </div>
    </section>
  </main>

  <!-- Footer -->
  <footer style="
    border-top:1px solid rgba(255,255,255,.12);
    padding:18px clamp(16px,5vw,48px); text-align:center; color:rgba(255,255,255,.72);
  ">
    © 2025 CARE Thematic Analytics • High contrast • Keyboard accessible
  </footer>

  <style>
    a:focus-visible, button:focus-visible { outline:3px solid var(--ring); outline-offset:3px; border-radius:14px; }
    @media (max-width:780px){ nav{display:none} }
    @media (prefers-reduced-motion: reduce){ *{transition:none !important; animation:none !important;} }
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
    # If user clicked the in-HTML button (?start=1), hide cover and go to login
    qp = st.query_params
    if qp.get("start") == "1":
        st.session_state.show_cover = False
        st.session_state.auth_mode = "login"
        # clear the query param to keep the URL clean
        try:
            st.query_params.clear()
        except Exception:
            # older Streamlit:
            st.experimental_set_query_params()

        st.rerun()

    # Render the full-bleed cover
    components.html(COVER_HTML, height=720, scrolling=False)

  

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





