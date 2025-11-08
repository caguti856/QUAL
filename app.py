
# main.py
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



# --- GLOBAL SESSION SETUP ---
# ✅ Only defined ONCE, here in main.py
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "show_cover" not in st.session_state:
    st.session_state["show_cover"] = True
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"
# --- Cover page content (HTML + CSS) ---
cover_page_html = """
<div style="font-family: 'Inter', sans-serif; background-color: #f6f7f8; height: 100vh; display: flex; flex-direction: column; justify-content: space-between; text-align: center; color: #101922; padding: 2rem 3rem;">
  <div>
    <h1 style="font-size: 2.8rem; font-weight: 900; margin-bottom: 0.75rem;">Streamline Your Data Workflow</h1>
    <p style="font-size: 1.1rem; color: #4b5563; max-width: 600px; margin: 0 auto 1.8rem; line-height: 1.5;">
      Powerful tools for data cleaning, database management, and insightful analytics.<br>
      Turn your raw data into actionable insights.
    </p>
  </div>

  <div style="display: flex; justify-content: center; gap: 1rem;">
    <div style="background-color: white; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 1.5rem; width: 240px;">
      <span style="font-size: 2rem; color: #1173d4;">🧹</span>
      <h3>Data Cleaning</h3>
      <p>Automatically detect and fix errors, duplicates, and inconsistencies.</p>
    </div>
    <div style="background-color: white; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 1.5rem; width: 240px;">
      <span style="font-size: 2rem; color: #1173d4;">🗄️</span>
      <h3>Database Management</h3>
      <p>Import, organize, and manage databases securely and efficiently.</p>
    </div>
    <div style="background-color: white; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 1.5rem; width: 240px;">
      <span style="font-size: 2rem; color: #1173d4;">📊</span>
      <h3>Analytics & Dashboards</h3>
      <p>Visualize your data and uncover meaningful insights.</p>
    </div>
  </div>

  <footer style="font-size: 0.85rem; color: #777; margin-top: 1rem;">© 2025 DataLens. All rights reserved.</footer>
</div>
"""

# --- Show cover page with native Streamlit button ---
def show_cover_page():
    # Render HTML
    components.html(cover_page_html, height=650, scrolling=False)
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
