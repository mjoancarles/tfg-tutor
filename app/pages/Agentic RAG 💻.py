import streamlit as st

st.set_page_config(
    page_title="Agentic RAG",
    page_icon=":desktop_computer:",
    layout="centered",
    initial_sidebar_state="collapsed", # To be changed if having filters!
)

# --- Enforce Login ---
if not st.session_state.get("logged_in", False):
    st.switch_page("Welcome.py")
    st.stop()