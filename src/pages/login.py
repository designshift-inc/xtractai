import streamlit as st

from utils.logging_config import setup_logging
logger = setup_logging()

def login_page():
    st.title("XtractAI - Demo")
    st.text("株式会社Design Shift")
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    # ログイン成功時にセッションステートを更新してログイン状態にする
    if login_button and login(username, password):
        st.session_state["logged_in"] = True
        st.rerun()  # ログイン後即座に画面を再描画

    # ログイン失敗時にエラーメッセージを表示
    elif login_button:
        st.error("Invalid username or password")

# シンプルなログイン機能
def login(username, password):
    return username == "demo" and password == "demo2024"