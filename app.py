import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="Thematic Analytics", layout="wide")
import importlib


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

# ---------------- COVER HTML ----------------
COVER_HTML = """
<div style="
  --care-orange:#EB7100;
  --bg:#F7F7F9;
  --panel:#FFFFFF;
  --text:#0F1222;
  --muted:#667085;
  --ring:rgba(235,113,0,.45);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  height:100vh; width:100vw; overflow:hidden;
  display:grid; grid-template-rows: auto 1fr auto;
  background:var(--bg);
  color:var(--text);
">
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

  <footer style="
    padding: 12px clamp(16px,4vw,40px); text-align:center;
    background:var(--panel); border-top:1px solid #E6E7EB; color:#5B616E; font-size:12px;
  ">
    © 2025 CARE Thematic Analytics
  </footer>
</div>
"""


# ---------------- SESSION STATE ----------------
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "show_cover" not in st.session_state:
    st.session_state["show_cover"] = True
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"

# ---------------- HELPERS ----------------
def show_cover_page():
    qp = st.query_params
    if qp.get("start") == "1":
        # User clicked "Get Started" -> go to login
        st.session_state.show_cover = False
        st.session_state.auth_mode = "login"
        try:
            st.query_params.clear()
        except Exception:
            st.experimental_set_query_params()
        st.rerun()

    # Only render cover (no other content on this run)
    components.html(COVER_HTML, height=720, scrolling=False)

@st.cache_resource(show_spinner=False)
def _lazy_import(module_name: str):
    return importlib.import_module(module_name)

def _render_tab(module_name: str, nice_name: str):
    st.markdown(
        f"""
        <div class="app-title">
          <h1>{nice_name}</h1>
          <span class="app-sub">Tap run inside the page when you’re ready.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        mod = _lazy_import(module_name)
        mod.main()
    except ModuleNotFoundError:
        st.error(f"Module `{module_name}.py` not found next to main.py.")
    except Exception as e:
        st.error(f"Failed to render **{nice_name}**: {type(e).__name__}: {e}")

# ---------------- MAIN ROUTER ----------------
def main():
    # If user is already logged in, NEVER show cover again
    if st.session_state.user_email:
        st.session_state.show_cover = False

    # 1) Show cover only when not logged in AND show_cover is True
    if st.session_state.show_cover and not st.session_state.user_email:
        show_cover_page()
        return

    # 2) If not logged in yet -> show login page
    if not st.session_state.user_email:
        try:
            login = _lazy_import("login")
            login.show_auth_page()
        except Exception as e:
            st.error(f"Login page error: {e}")
        return

    # 3) Logged in view: apply shell styling
    

    # Sidebar: only Logout
    with st.sidebar:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user_email = None
            st.session_state.show_cover = True  # show cover next time
            st.rerun()

    # Top nav (radio that behaves like tabs)
    st.markdown('<div class="nav-label">SECTIONS</div>', unsafe_allow_html=True)
    section = st.radio(
        "",
        [
            "Advisory",
            "Thought Leadership",
            "Growth Mindset",
            "Networking",
            "Influencing Relationship",
        ],
        horizontal=True,
        key="top_nav",
    )

    # Only run the selected module
    if section == "Advisory":
        _render_tab("advisory", "Advisory")
    elif section == "Thought Leadership":
        _render_tab("thoughtleadership", "Thought Leadership")
    elif section == "Growth Mindset":
        _render_tab("growthmindset", "Growth Mindset")
    elif section == "Networking":
        _render_tab("networking", "Networking")
    elif section == "Influencing Relationship":
        _render_tab("influencingrelationship", "Influencing Relationship")


if __name__ == "__main__":
    main()
