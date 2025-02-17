import streamlit as st

st.set_page_config(
    page_title="TFG Tutor Chatbot",
    page_icon=":wave:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Login Functionality ---
def login():
    st.title("Login")
    st.write("Please enter your credentials to continue.")
    
    # Get username and password input
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            # Rerun the app so that the welcome page shows up after login.
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

# --- Welcome / Navigation Page ---
def welcome_page():
    st.title("Welcome to TFG Tutor Chatbot")
    st.write("Please choose the approach you wish to test:")
    st.markdown(
        """
        - **RAG Approach:** Test the standard Retrieval-Augmented Generation (RAG) chain.
        - **Graph + Agent RAG Approach:** Test the agentic RAG approach using a graph-based knowledge source.
        """
    )
    
    # Navigation buttons: using switch_page (from streamlit-extras) if available
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("RAG Approach"):
            try:
                st.switch_page("pages/RAG 💻.py")
            except ImportError:
                st.info("Please use the sidebar menu to navigate to the RAG page.")
    
    with col2:
        if st.button("Agentic RAG + graph Approach"):
            st.info("This page is coming soon. Stay tuned!")
            # Agentic RAG 💻

def main():
    # If the user is not logged in, show the login page.
    if not st.session_state.get("logged_in", False):
        login()
        st.stop()  # Halt execution until login is successful.
    else:
        welcome_page()

if __name__ == "__main__":
    main()
