# auth.py
import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# ----- config: secrets in prod, .env locally -----
def _env(key: str, default: str = "") -> str:
    # prefer st.secrets in prod
    if key in st.secrets:
        return st.secrets[key]
    # fallback to .env for local dev
    load_dotenv()
    return os.getenv(key, default)

SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = _env("SUPABASE_SERVICE_ROLE_KEY")  # optional

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.stop()  # hard stop with a clear error in the UI
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
admin_supabase: Client | None = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    if SUPABASE_SERVICE_ROLE_KEY else None
)

# ----- session defaults -----
st.session_state.setdefault("user_email", None)
st.session_state.setdefault("auth_mode", "login")  # login | signup | forgot

# ----- auth API wrappers -----
def sign_up(email: str, password: str):
    return supabase.auth.sign_up({
        "email": email,
        "password": password,
        "options": {"emailRedirectTo": st.secrets.get("EMAIL_REDIRECT_TO", "")}
    })

def sign_in(email: str, password: str):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if res and res.user:
            st.session_state["user_email"] = res.user.email
            return res
    except Exception as e:
        st.error(f"Login failed: {e}")
    return None

def sign_out():
    try:
        supabase.auth.sign_out()
    finally:
        st.session_state["user_email"] = None

def admin_reset_password(email: str, new_password: str) -> bool:
    """Uses service role (if provided). Safer alternative is Supabase's RESET email flow."""
    if not admin_supabase:
        st.error("Admin password reset requires SUPABASE_SERVICE_ROLE_KEY.")
        return False
    try:
        # fetch all users (paginated in SDK v2; this returns a list-like)
        users = admin_supabase.auth.admin.list_users()
        user = next((u for u in users if getattr(u, "email", "").lower() == email.lower()), None)
        if not user:
            st.error("No user found with that email.")
            return False
        admin_supabase.auth.admin.update_user_by_id(user.id, {"password": new_password})
        return True
    except Exception as e:
        st.error(f"Reset failed: {e}")
        return False

# ----- UI -----
def show_auth_page():
    # Full screen glassy login view (minimal style for stability)
    st.markdown(
        "<h2 style='text-align:center;margin:0.5rem 0;'>Sign in to Thematic Analytics</h2>",
        unsafe_allow_html=True
    )

    mode = st.session_state.get("auth_mode", "login")

    if mode == "signup":
        st.write("Create your account")
        email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
        pw = st.text_input("Password", type="password", key="signup_pw", placeholder="••••••••")
        if st.button("Create Account", use_container_width=True):
            if not email or not pw:
                st.warning("Enter email and password.")
            else:
                res = sign_up(email, pw)
                if res and res.user:
                    st.success("Check your email to verify, then return to log in.")
        if st.button("Back to Login", use_container_width=True):
            st.session_state["auth_mode"] = "login"
            st.rerun()

    elif mode == "forgot":
        st.write("Reset your password (admin)")
        email = st.text_input("Email", key="forgot_email", placeholder="you@example.com")
        pw1 = st.text_input("New Password", type="password", key="forgot_pw1")
        pw2 = st.text_input("Confirm New Password", type="password", key="forgot_pw2")
        if st.button("Reset Password", use_container_width=True):
            if not email or not pw1 or not pw2:
                st.warning("Fill all fields.")
            elif pw1 != pw2:
                st.warning("Passwords do not match.")
            elif len(pw1) < 6:
                st.warning("Use at least 6 characters.")
            else:
                if admin_reset_password(email, pw1):
                    st.success("Password updated. Log in now.")
                    st.session_state["auth_mode"] = "login"
                    st.rerun()
        if st.button("Back to Login", use_container_width=True):
            st.session_state["auth_mode"] = "login"
            st.rerun()

    else:  # login
        email = st.text_input("Email", key="login_email", placeholder="you@example.com")
        pw = st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
        if st.button("Log In", use_container_width=True):
            if not email or not pw:
                st.warning("Enter email and password.")
            else:
                if sign_in(email, pw):
                    st.success("Logged in.")
                    st.rerun()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create Account", use_container_width=True):
                st.session_state["auth_mode"] = "signup"
                st.rerun()
        with c2:
            if st.button("Forgot Password?", use_container_width=True):
                st.session_state["auth_mode"] = "forgot"
                st.rerun()

def require_auth() -> bool:
    """Return True if logged in; otherwise render the auth page and return False."""
    if st.session_state.get("user_email"):
        return True
    # Center the auth box
    st.write("")
    col = st.columns([1, 1, 1])[1]
    with col:
        show_auth_page()
    return False
