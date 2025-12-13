# login.py
import os
import base64
import streamlit as st
from supabase import create_client, Client
st.markdown("""
    <style>
    /* Remove every possible top/bottom padding or margin */
    html, body, [data-testid="stAppViewContainer"], section.main, .block-container {
        padding: 0 !important;
        margin: 0 !important;
        height: 100vh !important;
        
    }

    /* Kill Streamlit header and toolbar spacing completely */
    
    }
    /* Force full height for layout columns */
    [data-testid="column"] {
        height: 100vh !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    /* Ensure left image fills completely */
    .image-container, .image-container img {
        top: 0 !important;
        left: 0 !important;
        bottom: 0 !important;
        width: 100% !important;
        height: 100vh !important;
        object-fit: cover !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Remove Streamlit default blue background (if any remains) */
    [data-testid="stAppViewContainer"] {
        background: none !important;
    }
    </style>
 """, unsafe_allow_html=True)


# ---------------- BASE STYLING ----------------
st.markdown("""
    <style>
    section.main > div, .block-container {
    padding:0 !important;
    margin:0 !important;
    max-width:100% !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none !important; }
    
    /* Form and input styling */
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
    .stButton>button:hover{
    transform:translateY(-2px);box-shadow:0 6px 18px rgba(17,115,212,.25);
    }
    /* Support link */
    .dl-support{text-align:center;margin-top:1.5rem;}
    .dl-support a{color:transparent;text-decoration:none;position:relative;}
    .dl-support a::after{
    content:'📧 Contact Support';color:#D3D3D3;font-size:.9rem;cursor:pointer;
    }
    .dl-support a:hover::after{text-decoration:underline;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- Glass blur background for login section --- */
[data-testid="column"]:nth-child(2) {
    position: relative !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    height: 100vh !important;
    background: rgba(255,255,255,0.12);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    backdrop-filter: blur(20px) saturate(180%);
    border-left: 1px solid rgba(255,255,255,0.15);
    box-shadow: inset 0 0 30px rgba(255,255,255,0.05);
}
/* --- Add soft glowing edge --- */
[data-testid="column"]:nth-child(2)::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: 6px;
    background: linear-gradient(180deg, #ffffff60, #ffffff20);
    box-shadow: 0 0 25px rgba(255,255,255,0.25);
}
/* --- Text readability + sizing --- */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-align: center !important;
    text-shadow: 0 0 10px rgba(0,0,0,0.4);
}
h1 { font-size: 2.2rem !important; margin-bottom: 0.4rem !important; }
h2 { font-size: 1.8rem !important; margin-bottom: 0.4rem !important; }
h3 { font-size: 1.4rem !important; }
p, label {
    color: #f8faff !important;
    text-shadow: 0 0 10px rgba(0,0,0,0.4);
    font-size: 1.2rem !important;
    line-height: 1.6 !important;
}
/* --- Input fields --- */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    font-size: 1.1rem !important;
}
.stTextInput > div > div > input::placeholder {
    color: #d9e6ff !important;
    opacity: 0.8 !important;
}
.stTextInput > div > div > input:focus {
    border-color: #66ccff !important;
    box-shadow: 0 0 14px rgba(79,195,255,0.5);
}
/* --- Buttons --- */
.stButton > button {
    background: linear-gradient(90deg, #4fc3ff, #007aff) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.5px !important;
    border-radius: 10px !important;
    padding: 1rem 0 !important;
    box-shadow: 0 0 25px rgba(79,195,255,0.35);
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 35px rgba(79,195,255,0.55);
}
@media (max-width: 900px) {
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.5rem !important; }
    p, label { font-size: 1.05rem !important; }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* 🔒 Lock full viewport size and stop scroll jump */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        width: 100vw !important;
        overflow: hidden !important;
    }
    /* 💎 Freeze Streamlit column layout height */
    [data-testid="stVerticalBlock"] {
        height: 100vh !important;
        overflow: hidden !important;
    }
    /* 🧩 Disable Streamlit’s rerender scroll reset */
    [data-testid="stSidebar"], [data-testid="stHeader"], footer {
        display: none !important;
    }
    /* 🌫️ Keep background stable on rerun */
    [data-testid="stAppViewContainer"] > div:first-child {
        position: fixed !important;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Lock the main block to one viewport and prevent scroll */
    .block-container, [data-testid="stVerticalBlock"] {
    height: 100vh !important;
        padding: 0 !important;
    margin: 0 !important;
    }
    /* Reserve space for alerts so height doesn't change on rerun */
    .stAlert {
    min-height: 56px;  /* adjust if your messages are taller */
    }
    /* Make form labels and helper text brighter on the glass */
    label, .stMarkdown p {
    color: #eaf3ff !important;
    text-shadow: 0 0 6px rgba(0,0,0,0.25);
    }
    /* Inputs readable on glass */
    .stTextInput > div > div > input {
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    }
    .stTextInput > div > div > input:focus {
    border-color: #4fc3ff !important;
    box-shadow: 0 0 10px rgba(79,195,255,0.45) !important;
    }
    /* Buttons pop */
    .stButton > button {
    background: linear-gradient(90deg, #4fc3ff, #007aff) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .9rem 0 !important;
    box-shadow: 0 0 18px rgba(79,195,255,.25) !important;
    }
    .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(79,195,255,.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* === FINAL LAYOUT STABILITY FIX === */

    /* Fix both halves to the viewport (no jump or scroll on rerun) */
    [data-testid="stAppViewContainer"] > div:first-child {
        position: fixed !important;
        top: 0; left: 0;
        width: 100vw !important;
        height: 100vh !important;
        background: #0b0b2a !important;
    }

    /* Left side image locked */
    .image-container, .image-container img {
        position: fixed !important;
        top: 0; left: 0;
        width: 55vw !important;
        height: 100vh !important;
        object-fit: cover !important;
        filter: brightness(78%);
        z-index: 1 !important;
    }

    /* Right glass panel locked next to image */
    [data-testid="column"]:nth-child(2) {
        position: fixed !important;
        right: 0; top: 0;
        width: 45vw !important;
        height: 100vh !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-left: 1px solid rgba(255,255,255,0.15) !important;
        box-shadow: inset 0 0 25px rgba(255,255,255,0.05);
        z-index: 2 !important;
    }

    /* Slight glow edge between halves */
    [data-testid="column"]:nth-child(2)::before {
        content: "";
        position: absolute;
        left: 0; top: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #ffffff40, #ffffff10);
        box-shadow: 0 0 20px rgba(255,255,255,0.2);
    }

    /* Prevent Streamlit default column gap on rerun */
    [data-testid="stHorizontalBlock"] {
        margin: 0 !important;
        gap: 0 !important;
    }

    /* Responsive layout: below 900px -> full width form */
    @media (max-width: 900px) {
        .image-container { display: none !important; }
        [data-testid="column"]:nth-child(2) {
            width: 100vw !important;
            border-left: none !important;
            background: rgba(255,255,255,0.1);
        }
    }
    </style>
""", unsafe_allow_html=True)
# -----------------------------
# Session defaults & helpers
# -----------------------------
def _ensure_defaults():
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None
    if "show_cover" not in st.session_state:
        st.session_state["show_cover"] = True



# -----------------------------
# Config: use Streamlit secrets
# -----------------------------
# Accept both flat and nested secrets:
#   - st.secrets["SUPABASE_URL"]
#   - st.secrets["SUPABASE_ANON_KEY"]
#   - st.secrets["SUPABASE_SERVICE_ROLE_KEY"]   (optional, for admin reset)
# Or nested:
#   - st.secrets["SUPABASE"]["url"], ["anon"], ["service_role"]
def _get_secret(*paths):
    # tries flat then nested dicts
    for key in paths:
        if key in st.secrets:
            return st.secrets.get(key)
    # nested common pattern
    if "SUPABASE" in st.secrets:
        node = st.secrets["SUPABASE"]
        for alt in ("url", "URL"):
            if alt in node:
                if "SUPABASE_URL" in paths:
                    return node[alt]
        for alt in ("anon", "ANON", "anon_key", "ANON_KEY"):
            if alt in node:
                if "SUPABASE_ANON_KEY" in paths:
                    return node[alt]
        for alt in ("service_role", "SERVICE_ROLE", "service_role_key", "SERVICE_ROLE_KEY"):
            if alt in node:
                if "SUPABASE_SERVICE_ROLE_KEY" in paths:
                    return node[alt]
    return None

SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = _get_secret("SUPABASE_ANON_KEY")
SERVICE_ROLE_KEY = _get_secret("SUPABASE_SERVICE_ROLE_KEY")  # optional

# -----------------------------
# Create Supabase clients
# -----------------------------
def _create_clients():
    sb = None
    admin = None

    if SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        except Exception as e:
            st.error(f"Failed to init anon client. Check SUPABASE_URL/ANON_KEY. Details: {e}")

    if SUPABASE_URL and SERVICE_ROLE_KEY:
        try:
            admin = create_client(SUPABASE_URL, SERVICE_ROLE_KEY)
        except Exception as e:
            st.warning(f"Admin client not initialized (SERVICE_ROLE_KEY). Password reset will be disabled. Details: {e}")

    return sb, admin

supabase, admin_supabase = _create_clients()



# -----------------------------
# Supabase wrapper helpers
# -----------------------------
def sign_up(email: str, password: str):
    if not supabase:
        st.error("Supabase anon client not initialized. Set secrets SUPABASE_URL and SUPABASE_ANON_KEY.")
        return None
    try:
        return supabase.auth.sign_up({
            "email": email,
            "password": password,
            # redirect can be your deployed app url + mode=signup
            "options": {"emailRedirectTo": st.secrets.get("EMAIL_REDIRECT_URL", "")}
        })
    except Exception as e:
        st.error(f"Sign up failed: {e}")
        return None

def sign_in(email: str, password: str):
    if not supabase:
        st.error("Supabase anon client not initialized. Set secrets SUPABASE_URL and SUPABASE_ANON_KEY.")
        return None
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if user and getattr(user, "user", None):
            st.session_state["user_email"] = user.user.email
            return user
        st.error("Invalid email or password.")
    except Exception as e:
        st.error(f"Login error: {e}")
    return None

def sign_out():
    if not supabase:
        st.session_state["user_email"] = None
        return
    try:
        supabase.auth.sign_out()
    finally:
        st.session_state["user_email"] = None

def admin_reset_password(email: str, new_password: str) -> bool:
    """Reset a user password with service role (requires SERVICE_ROLE_KEY)."""
    if not admin_supabase:
        st.error("Admin client not initialized. Add SUPABASE_SERVICE_ROLE_KEY in secrets to enable password reset.")
        return False
    try:
        # list users (SDK returns a dict-like or object depending on version)
        users = admin_supabase.auth.admin.list_users()
        # unify iteration
        items = []
        if isinstance(users, dict) and "users" in users:
            items = users["users"]
        elif isinstance(users, (list, tuple)):
            items = users
        else:
            # fallback: try attribute
            items = getattr(users, "users", [])

        user = None
        for u in items:
            # u may be dict or object
            u_email = u.get("email") if isinstance(u, dict) else getattr(u, "email", None)
            if u_email and u_email.lower() == email.lower():
                user = u
                break

        if not user:
            st.error("❌ No user found with that email.")
            return False

        user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
        if not user_id:
            st.error("Could not read user id from admin list response.")
            return False

        admin_supabase.auth.admin.update_user_by_id(user_id, {"password": new_password})
        st.success(f"✅ Password for **{email}** updated. You can log in with the new password.")
        return True
    except Exception as e:
        st.error(f"Password reset failed: {e}")
        return False

# -----------------------------
# UI bits
# -----------------------------
def _left_image_block():
    # Try to show a left-side image; gracefully handle missing file
    col_bg = st.container()
    try:
        with open("IMAGES/background.jpg", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        col_bg.markdown(
            f"""
            <div class="image-container" style="position:fixed;left:0;top:0;width:55vw;height:100vh;overflow:hidden">
              <img src="data:image/jpg;base64,{encoded}" style="width:100%;height:100%;object-fit:cover;filter:brightness(78%)" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        # no image — just a solid color
        col_bg.markdown(
            """
            <div style="position:fixed;left:0;top:0;width:55vw;height:100vh;background:#0b0b2a"></div>
            """,
            unsafe_allow_html=True,
        )

def show_auth_page():
    _ensure_defaults()

    # Handle query params for signup/verify/reset deep-links
    qp = st.query_params
    mode = qp.get("mode", "")

    if mode == "reset":
        if st.session_state.get("auth_mode") != "reset_password":
            st.session_state["auth_mode"] = "reset_password"
            st.session_state["is_reset_link"] = True
            st.rerun()
    elif st.session_state.get("is_reset_link"):
        st.session_state["auth_mode"] = "reset_password"
    elif mode == "signup":
        st.session_state["auth_mode"] = "login"
        st.success("✅ Email confirmed! You can now log in.")
        st.experimental_set_query_params()

    # Layout
    col1, col2 = st.columns([0.47, 0.37], gap="small")

    with col1:
        _left_image_block()

    with col2:
        st.markdown("<div style='width:92%; max-width:480px;'>", unsafe_allow_html=True)

        # -------------------------
        # AUTH LOGIC
        # -------------------------
        auth_mode = st.session_state.get("auth_mode", "login")

        if auth_mode == "signup":
            st.header("Create Account")
            st.write("Join and unlock smart data management.")
            email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="signup_password", placeholder="••••••••")

            if st.button("Sign Up"):
                if email and password:
                    user = sign_up(email, password)
                    if user and getattr(user, "user", None):
                        st.success("🎉 Account created! Check your email to verify before logging in.")
                    else:
                        st.error("Sign up failed. Try again.")
                else:
                    st.warning("Please enter all fields.")

            if st.button("Back to Login"):
                st.session_state["auth_mode"] = "login"
                st.rerun()

        elif auth_mode in ("forgot", "reset_password"):
            st.header("Reset Your Password")
            st.write("Enter your email and a new password.")

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
                        st.success("✅ Password updated! You can now log in.")
                        st.session_state["auth_mode"] = "login"
                        st.rerun()

            if st.button("Back to Login"):
                st.session_state["auth_mode"] = "login"
                st.rerun()

        else:
            st.header("Welcome Back")
            st.write("Log in to access your workspace.")
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
                    st.session_state["auth_mode"] = "signup"
                    st.rerun()
            with c2:
                if st.button("Forgot Password?"):
                    st.session_state["auth_mode"] = "forgot"
                    st.rerun()

       

        st.markdown("""
        <div class="dl-support" style="text-align:center; margin-top:1.2rem;">
            <a href="mailto:caguti856@gmail.com" target="_blank"
               style="color:#4fc3ff; font-weight:600; font-size:0.95rem; text-decoration:none;">
               📧 Contact Support
            </a>
        </div>
        """, unsafe_allow_html=True)

def show_main_app():
    st.title("Welcome to the Dashboard")
    st.success(f"Logged in as {st.session_state.get('user_email')}")
    if st.button("Log Out"):
        sign_out()
        st.rerun()

def main():
    _ensure_defaults()
    if not st.session_state.get("user_email"):
        show_auth_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
