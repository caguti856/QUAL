 
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


# --- Global no-scroll for Streamlit chrome (keep if you haven't already) ---
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"]{
  height:100vh !important; width:100vw !important; margin:0 !important; padding:0 !important;
  overflow:hidden !important;
}
.block-container, section.main{ padding:0 !important; margin:0 !important; }
header, [data-testid="stHeader"], [data-testid="stToolbar"], footer{ display:none !important; }
</style>
""", unsafe_allow_html=True)

# --- No-scroll COVER (white/grey background, CARE orange accents) ---
COVER_HTML = """
<div style="
  --care-orange:#EB7100;
  --bg:#F7F7F9;     /* light page */
  --panel:#FFFFFF;  /* header/footer panels */
  --text:#0F1222;
  --muted:#667085;
  --ring:rgba(235,113,0,.45);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;

  /* full viewport, absolutely no scroll */
  height:100vh; width:100vw; overflow:hidden;
  display:grid; grid-template-rows: auto 1fr auto;
  background:var(--bg);
  color:var(--text);
">

  <!-- Header (sticky look but we don't scroll anyway) -->
  <header style="
    display:flex; align-items:center; justify-content:space-between;
    padding: 14px clamp(16px,4vw,40px);
    background:var(--panel); border-bottom:1px solid #E6E7EB;
  ">
    <div style="display:flex; align-items:center; gap:.75rem;">
      <img src="https://brand.care.org/wp-content/uploads/2017/08/Orange-Horizontal-300x97.png"
           alt="CARE" style="height:28px; display:block"/>
     
    </div>
    <div style="opacity:.85; font-size:.95rem; color:#6B7280;">Thematic Analytics</div>
  </header>

  <!-- Hero (centered; uses clamp to ALWAYS fit) -->
  <main style="display:grid; place-items:center; padding: clamp(8px, 2vw, 24px);">
    <section style="text-align:center; max-width:1050px; width:100%;">
      <h1 style="
        margin:0 0 .6rem 0;
        font-size: clamp(28px, 5.2vw, 54px);
        line-height:1.08; letter-spacing:-.02em; font-weight:900;
      ">
        Consistent scoring. Credible insights. Bigger impact.
      </h1>
      <p style="
        margin:0 auto 18px; max-width:780px;
        color:var(--muted); font-size:clamp(14px, 2.2vw, 18px);
      ">
        Turn qualitative feedback into defensible, program-ready evidence.
      </p>

      <!-- CTA -->
      <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:10px;">
        <a role="button" href="?start=1" style="
          display:inline-flex; align-items:center; gap:.55rem;
          background:var(--care-orange); color:#fff; font-weight:800;
          padding:12px 20px; border-radius:999px; text-decoration:none;
          box-shadow:0 10px 22px rgba(235,113,0,.28);
        ">🚀 Get Started</a>
      </div>
    </section>
  </main>

  <!-- Footer (thin, within viewport) -->
  <footer style="
    padding: 12px clamp(16px,4vw,40px); text-align:center;
    background:var(--panel); border-top:1px solid #E6E7EB; color:#5B616E; font-size:12px;
  ">
    © 2025 CARE Thematic Analytics
  </footer>

  <style>
    a:focus-visible{ outline:3px solid var(--ring); outline-offset:3px; border-radius:999px; }
    @media (max-width:900px){
      /* one column tiles on small screens, still fits */
      main section div[style*='grid-template-columns:repeat(3']{ grid-template-columns:1fr !important; }
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


def show_cover_page():
    qp = st.query_params
    if qp.get("start") == "1":
        st.session_state.show_cover = False
        st.session_state.auth_mode = "login"
        try:
            st.query_params.clear()
        except Exception:
            st.experimental_set_query_params()
        st.rerun()

    # exactly one viewport tall, no internal scroll
    components.html(COVER_HTML, height=700, scrolling=False)


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





