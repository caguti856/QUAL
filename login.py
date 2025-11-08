# login.py — no dotenv required
import streamlit as st
from supabase import create_client, Client
import os
import base64

# ---- at the top of login.py, replace your current init/creds section ----
import streamlit as st
from supabase import create_client, Client
import os
import base64

# Read from Streamlit secrets first, fall back to env
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = (
st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# Create a PUBLIC client for normal auth (signup/login)
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        st.error(f"Failed to init Supabase client (anon): {e}")

# Create a separate ADMIN client (optional) for password resets
admin_supabase: Client | None = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        admin_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as e:
        st.error(f"Failed to init Supabase admin client: {e}")

# Early diagnostics shown in a collapsible block (optional)
with st.expander("🔧 Supabase diagnostics", expanded=False):
    st.write("URL set:", bool(SUPABASE_URL))
    st.write("Anon key set:", bool(SUPABASE_ANON_KEY))
    st.write("Service role key set:", bool(SUPABASE_SERVICE_ROLE_KEY))
    if SUPABASE_URL and not SUPABASE_URL.startswith("https://"):
        st.warning("SUPABASE_URL should start with https://")

# Ensure session keys exist early
st.session_state.setdefault("user_email", None)
st.session_state.setdefault("auth_mode", "login")

# =========================
# Global CSS (your styles)
# =========================
st.markdown("""
    <style>
    /* Remove every possible top/bottom padding or margin */
    html, body, [data-testid="stAppViewContainer"], section.main, .block-container {
        padding: 0 !important;
        margin: 0 !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    header, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
    [data-testid="column"] { height: 100vh !important; padding: 0 !important; margin: 0 !important; }
    .image-container, .image-container img {
        top: 0 !important; left: 0 !important; bottom: 0 !important;
        width: 100% !important; height: 100vh !important; object-fit: cover !important;
        margin: 0 !important; padding: 0 !important;
    }
    [data-testid="stAppViewContainer"] { background: none !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    section.main > div, .block-container {
        padding:0 !important; margin:0 !important; max-width:100% !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none !important; }
    [data-testid="stAppViewContainer"] { background:#0b0b2a; overflow:hidden; }
    h2{color:#0f2748 !important;font-weight:800 !important;text-align:center;margin-bottom:.4rem !important;}
    p{color:#475569 !important;text-align:center;margin-bottom:1.4rem !important;}
    .stTextInput>div>div>input{
        border-radius:10px;border:1px solid #cbd5e1;padding:.7rem .9rem;font-size:.95rem;background:#fff;
    }
    .stTextInput>div>div>input:focus{
        border-color:#D3D3D3;box-shadow:0 0 0 3px rgba(17,115,212,.15);
    }
    .stButton>button{
        width:100%;background:linear-gradient(90deg,#D3D3D3,#0f63b5);
        color:#fff;border:none;padding:.9rem 0;border-radius:10px;
        font-weight:600;font-size:.95rem;margin-top:1rem;cursor:pointer;transition:all .25s ease;
    }
    .stButton>button:hover{ transform:translateY(-2px);box-shadow:0 6px 18px rgba(17,115,212,.25); }
    .dl-support{text-align:center;margin-top:1.5rem;}
    .dl-support a{color:transparent;text-decoration:none;position:relative;}
    .dl-support a::after{ content:'📧 Contact Support';color:#D3D3D3;font-size:.9rem;cursor:pointer; }
    .dl-support a:hover::after{text-decoration:underline;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    [data-testid="column"]:nth-child(2) {
        position: relative !important; display: flex !important;
        justify-content: center !important; align-items: center !important;
        height: 100vh !important; background: rgba(255,255,255,0.12);
        -webkit-backdrop-filter: blur(20px) saturate(180%); backdrop-filter: blur(20px) saturate(180%);
        border-left: 1px solid rgba(255,255,255,0.15); box-shadow: inset 0 0 30px rgba(255,255,255,0.05);
    }
    [data-testid="column"]:nth-child(2)::before {
        content: ""; position: absolute; left: 0; top: 0; height: 100%; width: 6px;
        background: linear-gradient(180deg, #ffffff60, #ffffff20);
        box-shadow: 0 0 25px rgba(255,255,255,0.25);
    }
    h1, h2, h3 {
        color: #ffffff !important; font-weight: 800 !important; text-align: center !important;
        text-shadow: 0 0 10px rgba(0,0,0,0.4);
    }
    h1 { font-size: 2.2rem !important; margin-bottom: 0.4rem !important; }
    h2 { font-size: 1.8rem !important; margin-bottom: 0.4rem !important; }
    h3 { font-size: 1.4rem !important; }
    p, label {
        color: #f8faff !important; text-shadow: 0 0 10px rgba(0,0,0,0.4);
        font-size: 1.2rem !important; line-height: 1.6 !important;
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.15) !important; color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.35) !important; border-radius: 10px !important;
        padding: 1rem 1.2rem !important; font-size: 1.1rem !important;
    }
    .stTextInput > div > div > input::placeholder { color: #d9e6ff !important; opacity: 0.8 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #66ccff !important; box-shadow: 0 0 14px rgba(79,195,255,0.5);
    }
    .stButton > button {
        background: linear-gradient(90deg, #4fc3ff, #007aff) !important; border: none !important; color: #ffffff !important;
        font-weight: 700 !important; font-size: 1.1rem !important; letter-spacing: 0.5px !important;
        border-radius: 10px !important; padding: 1rem 0 !important; box-shadow: 0 0 25px rgba(79,195,255,0.35);
    }
    .stButton > button:hover { transform: scale(1.05); box-shadow: 0 0 35px rgba(79,195,255,0.55); }
    @media (max-width: 900px) {
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.5rem !important; }
        p, label { font-size: 1.05rem !important; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { height: 100vh !important; width: 100vw !important; overflow: hidden !important; }
    [data-testid="stVerticalBlock"] { height: 100vh !important; overflow: hidden !important; }
    [data-testid="stSidebar"], [data-testid="stHeader"], footer { display: none !important; }
    [data-testid="stAppViewContainer"] > div:first-child {
        position: fixed !important; top: 0; left: 0; width: 100vw; height: 100vh;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .block-container, [data-testid="stVerticalBlock"] {
        height: 100vh !important; overflow: hidden !important; padding: 0 !important; margin: 0 !important;
    }
    .stAlert { min-height: 56px; }
    label, .stMarkdown p { color: #eaf3ff !important; text-shadow: 0 0 6px rgba(0,0,0,0.25); }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.12) !important; color: #fff !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4fc3ff !important; box-shadow: 0 0 10px rgba(79,195,255,0.45) !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4fc3ff, #007aff) !important; color: #fff !important;
        font-weight: 700 !important; border: none !important; border-radius: 10px !important;
        padding: .9rem 0 !important; box-shadow: 0 0 18px rgba(79,195,255,.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 10px 22px rgba(79,195,255,.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > div:first-child {
        position: fixed !important; top: 0; left: 0; width: 100vw !important; height: 100vh !important;
        overflow: hidden !important; background: #0b0b2a !important;
    }
    .image-container, .image-container img {
        position: fixed !important; top: 0; left: 0; width: 55vw !important; height: 100vh !important;
        object-fit: cover !important; filter: brightness(78%); z-index: 1 !important;
    }
    [data-testid="column"]:nth-child(2) {
        position: fixed !important; right: 0; top: 0; width: 45vw !important; height: 100vh !important;
        display: flex !important; justify-content: center !important; align-items: center !important;
        background: rgba(255, 255, 255, 0.08) !important; backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important; border-left: 1px solid rgba(255,255,255,0.15) !important;
        box-shadow: inset 0 0 25px rgba(255,255,255,0.05); z-index: 2 !important;
    }
    [data-testid="column"]:nth-child(2)::before {
        content: ""; position: absolute; left: 0; top: 0; width: 6px; height: 100%;
        background: linear-gradient(180deg, #ffffff40, #ffffff10); box-shadow: 0 0 20px rgba(255,255,255,0.2);
    }
    [data-testid="stHorizontalBlock"] { margin: 0 !important; gap: 0 !important; }
    @media (max-width: 900px) {
        .image-container { display: none !important; }
        [data-testid="column"]:nth-child(2) { width: 100vw !important; border-left: none !important; background: rgba(255,255,255,0.1); }
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Supabase helpers
# =========================

def sign_up(email: str, password: str):
    if not supabase:
        st.error("Supabase not initialized (anon client). Check SUPABASE_URL and SUPABASE_ANON_KEY in secrets.")
        return None
    return supabase.auth.sign_up({
        "email": email,
        "password": password,
        "options": {
            "emailRedirectTo": "https://ven36ayip6wnsgdphrjagb.streamlit.app/?mode=signup"
        }
    })

def sign_in(email: str, password: str):
    if not supabase:
        st.error("Supabase not initialized (anon client).")
        return None
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if user and user.user:
            st.session_state.user_email = user.user.email
            return user
        st.error("Invalid email or password.")
    except Exception as e:
        st.error(f"Login failed: {e}")
    return None

def admin_reset_password(email: str, new_password: str) -> bool:
    if not admin_supabase:
        st.error("Admin client not available. Add SUPABASE_SERVICE_ROLE_KEY to secrets.")
        return False
    try:
        # Supabase Python SDK 2.x: list_users returns object with .users
        listing = admin_supabase.auth.admin.list_users()
        users = getattr(listing, "users", None) or listing
        user = next((u for u in users if getattr(u, "email", "").lower() == email.lower()), None)
        if not user:
            st.error("❌ No user found with that email.")
            return False
        admin_supabase.auth.admin.update_user_by_id(user.id, {"password": new_password})
        st.success(f"✅ Password for **{email}** has been updated.")
        return True
    except Exception as e:
        st.error(f"⚠️ Password reset failed: {e}")
        return False


def sign_out():
    if not supabase:
        st.session_state.user_email = None
        return
    try:
        supabase.auth.sign_out()
    finally:
        st.session_state.user_email = None



# =========================
# UI
# =========================
def _left_image_block():
    # Safely load background image (optional)
    try:
        with open("IMAGES/background.jpg", "rb") as f:
            img_data = f.read()
        encoded = base64.b64encode(img_data).decode()
        st.markdown(f"""
            <div class="image-container">
                <img src="data:image/jpg;base64,{encoded}" alt="Background">
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        # Fallback: gradient block
        st.markdown("""
            <div class="image-container" style="
                background: radial-gradient(1200px 800px at 20% 30%, #243b55, #141e30);
            "></div>
        """, unsafe_allow_html=True)

def show_auth_page():
    # Handle query params for signup/verify/reset deep-links
    qp = st.query_params
    mode = qp.get("mode", "")

    if mode == "reset":
        if st.session_state.get("auth_mode") != "reset_password":
            st.session_state.auth_mode = "reset_password"
            st.session_state["is_reset_link"] = True
            st.rerun()
    elif st.session_state.get("is_reset_link"):
        st.session_state.auth_mode = "reset_password"
    elif mode == "signup":
        st.session_state.auth_mode = "login"
        st.success("✅ Email confirmed! You can now log in.")
        st.experimental_set_query_params()

    # Layout
    col1, col2 = st.columns([0.47, 0.37], gap="small")

    with col1:
        _left_image_block()

    with col2:
        st.markdown("<div style='width:90%; max-width:450px;'>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "signup":
            st.header("Create Account")
            st.write("Join DataLens and unlock smart data management.")
            email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="signup_password", placeholder="••••••••")

            if st.button("Sign Up"):
                if email and password:
                    user = sign_up(email, password)
                    if user and getattr(user, "user", None):
                        st.success("🎉 Account created successfully!")
                        st.info("📩 Check your email and verify your account before logging in.")
                    else:
                        st.error("Sign up failed. Try again.")
                else:
                    st.warning("Please enter all fields.")

            if st.button("Back to Login"):
                st.session_state.auth_mode = "login"
                st.rerun()

        elif st.session_state.auth_mode in ("forgot", "reset_password"):
            st.header("Reset Your Password")
            st.write("Enter your email and a new password. You’ll be able to log in immediately after reset.")

            email = st.text_input("Email", key="reset_email", placeholder="you@example.com")
            new_password = st.text_input("New Password", type="password", key="reset_pw", placeholder="••••••••")
            confirm_pw = st.text_input("Confirm New Password", type="password", key="reset_confirm", placeholder="••••••••")

            if st.button("Reset Password"):
                if not email or not new_password or not confirm_pw:
                    st.warning("Please fill in all fields.")
                elif new_password != confirm_pw:
                    st.warning("Passwords do not match.")
                elif len(new_password) < 6:
                    st.warning("Password must be at least 6 characters long.")
                else:
                    if admin_reset_password(email, new_password):
                        st.success("✅ Password updated successfully! You can now log in.")
                        st.session_state.auth_mode = "login"
                        st.rerun()

            if st.button("Back to Login"):
                st.session_state.auth_mode = "login"
                st.rerun()

        else:
            st.header("Welcome Back")
            st.write("Log in to access your Qual workspace.")
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")

            if st.button("Log In"):
                if email and password:
                    user = sign_in(email, password)
                    if user and getattr(user, "user", None):
                        st.session_state["user_email"] = user.user.email.lower().strip()
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("Login failed. Check credentials.")
                else:
                    st.warning("Enter your email and password.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Create Account"):
                    st.session_state.auth_mode = "signup"
                    st.rerun()
            with c2:
                if st.button("Forgot Password?"):
                    st.session_state.auth_mode = "forgot"
                    st.rerun()

        st.markdown("""
        <div class="dl-support" style="text-align:center; margin-top:1.5rem;">
            <a href="mailto:caguti856@gmail.com" target="_blank"
               style="color:#4fc3ff; font-weight:600; font-size:0.95rem; text-decoration:none;">
               📧 Contact Support
            </a>
        </div>
        """, unsafe_allow_html=True)

        # --- 👇 Add this tiny self-test expander here ---
        with st.expander("Run a connection test"):
            if st.button("Test Supabase auth ping"):
                try:
                    # This just checks the client exists and can be called.
                    # It doesn't require a logged-in session.
                    _ = getattr(supabase.auth, "get_user", None)
                    if supabase is None:
                        st.error("Anon client not initialized.")
                    else:
                        # If SDK v2: get_session() may exist; if not, just accessing auth is fine.
                        st.success("✅ Supabase anon client is initialized and callable.")
                except Exception as e:
                    st.error(f"Anon client call failed: {e}")


def show_main_app():
    st.title("Welcome to the Dashboard")
    st.success(f"Logged in as {st.session_state.user_email}")
    if st.button("Log Out"):
        sign_out()
        st.rerun()

def main():
    if "user_email" not in st.session_state or not st.session_state.user_email:
        show_auth_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
