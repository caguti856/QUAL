# login.py
from __future__ import annotations

import base64
import os
import streamlit as st
from supabase import create_client, Client

# --------------------------
# CONFIG via Streamlit secrets (no dotenv)
# --------------------------
SUPABASE_URL = (st.secrets.get("SUPABASE_URL") or "").strip()
SUPABASE_ANON_KEY = (st.secrets.get("SUPABASE_ANON_KEY") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()  # optional if you use admin reset

# Lazy singletons to avoid crashing at import time
_supabase: Client | None = None
_admin: Client | None = None

def get_clients() -> tuple[Client | None, Client | None]:
    """Create clients only when needed, with clear UI errors if secrets are missing."""
    global _supabase, _admin

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Supabase not configured: set SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets.")
        return None, None

    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    if SUPABASE_SERVICE_ROLE_KEY and _admin is None:
        _admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    return _supabase, _admin


# --------------------------
# Global session defaults
# --------------------------
st.session_state.setdefault("user_email", None)
st.session_state.setdefault("auth_mode", "login")
st.session_state.setdefault("show_cover", True)


# --------------------------
# Minimal CSS you already had (keep as-is if you like)
# --------------------------
# (Shortened; keep your original long CSS if you want the full styling.)
st.markdown("""
<style>
[data-testid="stSidebar"], header, footer { display:none !important; }
/* ... your existing CSS blocks can stay here ... */
</style>
""", unsafe_allow_html=True)


# --------------------------
# Supabase auth helpers
# --------------------------
def sign_up(email: str, password: str):
    supa, _ = get_clients()
    if not supa: 
        return None
    return supa.auth.sign_up({
        "email": email,
        "password": password,
        "options": {
            # IMPORTANT: set this to YOUR deployed app URL
            "emailRedirectTo": "https://YOUR-APP-NAME.streamlit.app/?mode=signup"
        }
    })

def sign_in(email: str, password: str):
    supa, _ = get_clients()
    if not supa: 
        return None
    try:
        return supa.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        st.error(f"Login error: {e}")
        return None

def sign_out():
    supa, _ = get_clients()
    if supa:
        try:
            supa.auth.sign_out()
        except Exception:
            pass
    st.session_state["user_email"] = None

def admin_reset_password(email: str, new_password: str) -> bool:
    """Requires SUPABASE_SERVICE_ROLE_KEY in secrets. Otherwise, hide the reset UI."""
    _, admin = get_clients()
    if not SUPABASE_SERVICE_ROLE_KEY:
        st.error("Admin reset not configured: add SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets.")
        return False
    if not admin:
        return False
    try:
        # SDK responses vary: handle both shapes
        users_resp = admin.auth.admin.list_users()
        user_list = getattr(users_resp, "data", users_resp)
        if hasattr(user_list, "users"):
            user_list = user_list.users

        # Each user object should have .email/.id
        target = next((u for u in user_list if getattr(u, "email", "").lower() == email.lower()), None)
        if not target:
            st.error("No user found with that email.")
            return False

        admin.auth.admin.update_user_by_id(target.id, {"password": new_password})
        st.success(f"Password for {email} updated.")
        return True
    except Exception as e:
        st.error(f"Password reset failed: {e}")
        return False


# --------------------------
# UI helpers
# --------------------------
def _left_image_block():
    # Optional image on the left; safely handle missing file
    img_path = "IMAGES/background.jpg"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div class="image-container" style="position:fixed;left:0;top:0;width:55vw;height:100vh;">
            <img src="data:image/jpg;base64,{encoded}" style="width:100%;height:100%;object-fit:cover;filter:brightness(78%)" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='position:fixed;left:0;top:0;width:55vw;height:100vh;background:#0b0b2a'></div>",
            unsafe_allow_html=True
        )


# --------------------------
# The auth page (signup/login/reset)
# --------------------------
def show_auth_page():
    # Handle query params for signup/verify/reset
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
        # clear query param to prevent loops
        try:
            st.experimental_set_query_params()
        except Exception:
            pass

    col1, col2 = st.columns([0.47, 0.37], gap="small")

    with col1:
        _left_image_block()

    with col2:
        st.markdown("<div style='width:90%; max-width:450px;'>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "signup":
            st.header("Create Account")
            st.write("Join Thematic Analytics.")
            email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="signup_password", placeholder="••••••••")

            if st.button("Sign Up"):
                if email and password:
                    user = sign_up(email, password)
                    if user and getattr(user, "user", None):
                        st.success("🎉 Account created! Check your email to verify, then log in.")
                    else:
                        st.error("Sign up failed. Try again.")
                else:
                    st.warning("Please enter all fields.")

            if st.button("Back to Login"):
                st.session_state.auth_mode = "login"
                st.rerun()

        elif st.session_state.auth_mode in ("forgot", "reset_password"):
            st.header("Reset Your Password")
            st.write("Enter your email and a new password.")
            email = st.text_input("Email", key="reset_email", placeholder="you@example.com")
            new_pw = st.text_input("New Password", type="password", key="reset_pw", placeholder="••••••••")
            confirm_pw = st.text_input("Confirm New Password", type="password", key="reset_confirm", placeholder="••••••••")

            if st.button("Reset Password"):
                if not email or not new_pw or not confirm_pw:
                    st.warning("Please fill in all fields.")
                elif new_pw != confirm_pw:
                    st.warning("Passwords do not match.")
                elif len(new_pw) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    if admin_reset_password(email, new_pw):
                        st.success("✅ Password updated! Log in now.")
                        st.session_state.auth_mode = "login"
                        st.rerun()

            if st.button("Back to Login"):
                st.session_state.auth_mode = "login"
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
                    st.session_state.auth_mode = "signup"
                    st.rerun()
            with c2:
                # Show reset option only if service role configured
                if SUPABASE_SERVICE_ROLE_KEY:
                    if st.button("Forgot Password?"):
                        st.session_state.auth_mode = "forgot"
                        st.rerun()
                else:
                    st.caption("Admin reset is disabled (no service role key).")

        # ---- Support link
        st.markdown("""
        <div class="dl-support" style="text-align:center; margin-top:1.5rem;">
            <a href="mailto:caguti856@gmail.com" target="_blank"
               style="color:#4fc3ff; font-weight:600; font-size:0.95rem; text-decoration:none;">
               📧 Contact Support
            </a>
        </div>
        """, unsafe_allow_html=True)

        # ---- Quick connection test
        with st.expander("Run a connection test"):
            if st.button("Test Supabase auth ping"):
                supa, admin = get_clients()
                if not supa:
                    st.error("Anon client NOT initialized.")
                else:
                    st.success("✅ Anon client is initialized.")
                if SUPABASE_SERVICE_ROLE_KEY:
                    st.info("Service role key present.")
                    if admin:
                        st.success("🔐 Admin client is initialized.")
                    else:
                        st.warning("Admin client failed to initialize.")


# --------------------------
# Simple protected page (optional)
# --------------------------
def show_main_app():
    st.title("Welcome to the Dashboard")
    st.success(f"Logged in as {st.session_state.user_email}")
    if st.button("Log Out"):
        sign_out()
        st.rerun()


# --------------------------
# Public main entry if you run login.py directly
# --------------------------
def main():
    if not st.session_state.get("user_email"):
        show_auth_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
