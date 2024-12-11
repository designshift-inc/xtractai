import streamlit as st
from pages.login import login_page
from pages.xtract import xtract_page

def main():
    # セッションステートにログイン状態を保持
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if not st.session_state["logged_in"]:
        login_page()
    else:
        xtract_page()

if __name__ == "__main__":
    main()