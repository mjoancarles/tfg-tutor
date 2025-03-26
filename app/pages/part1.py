from langfuse.callback import CallbackHandler
import os
# https://python.langchain.com/docs/tutorials/rag/
import streamlit as st
import logging
from utils import * # Import the utils.py functions
from langchain_core.documents import Document
from langchain_core.prompts.base import format_document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(
    page_title="TFG Tutor Chatbot",
    page_icon=":desktop_computer:",
    layout="centered",
    initial_sidebar_state="collapsed", # To be changed if having filters!
)

# --- Enforce Login ---
_ ='''if not st.session_state.get("logged_in", False):
    st.switch_page("Welcome.py")
    st.stop()'''

# --- App Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

check_connections()
llm = init_llm()
embeddings = init_embeddings()
app_name = "rag-app"
langfuse_handler = CallbackHandler()

# Initialize vector store with Weaviate and custom embeddings
# https://python.langchain.com/docs/integrations/vectorstores/qdrant/
vector_store = get_qdrant_vector_store(embeddings)

def initialize_app():

    template = """
    General Info:
    You are an assistant answering questions about TFGs (Final Degree Projects by Catalan students) and CVC scientific publications. The CVC is a non-profit research center established in 1995 by the Generalitat de Catalunya and the UAB, focusing on computer vision research and collaborating on TFGs.

    Instructions:

    -When Responding:
    Only include retrieved documents if they are relevant to the query. If no relevant documents are found, answer directly.
    -Document Formatting:
    For Publications:
        Title: (publication title)
        - Authors: (comma-separated list)
        - Abstract: (summarized abstract)
        - Published in: (year)
        - Link: (if available)
    For TFGs:
        Title: (TFG title)
        - Authors: (comma-separated list)
        - Abstract: (summarized abstract)
        - Delivered in: (year)
        - Link: (if available)
    Exclude any field that is NULL/EMPTY. Show only the top 5 most relevant results by default (unless the user requests more). After listing, include a brief explanation and comparison of the documents, highlighting key details in bold.
    Direct Answers:
    If the query does not require listing documents, answer directly without showing retrieved documents. Always answer in english.
    ------------------------------
    Retrieved documents: {context}
        """
    prompt = PromptTemplate.from_template(template)

    document_prompt = PromptTemplate.from_template(
                "Title: {title}\n"
                "Type: {type}\n"
                "Abstract: {page_content}\n"
                "Authors: {authors}\n"
                "Year: {year}\n"
                "Keywords: {keywords}\n"
                "Link: {link}\n"
    )

    # https://python.langchain.com/docs/tutorials/rag/
    class State(TypedDict):
        messages: List[BaseMessage]
        context: List[Document]

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["messages"][-1].content)
        return {"context": retrieved_docs}

    def generate(state: State):
        messages = state["messages"]
        print(messages)
        #docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        docs_content = "\n\n".join(format_document(doc, document_prompt) for doc in state["context"])
        system_prompt = prompt.invoke({"context": docs_content}).to_string()
        #print(system_prompt)
        messages = [SystemMessage(system_prompt)] + messages
        print(messages)
        response = llm.invoke(messages)
        #print(type(response)) # <class 'langchain_core.messages.ai.AIMessage'>
        return {"messages": [response]}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("generate", END)
    graph = graph_builder.compile()
    return graph

def main():
    # Ensure 'graph' is created only once per session
    if "graph" not in st.session_state:
        st.session_state.graph = initialize_app()

    # We’ll use a local variable for convenience
    graph = st.session_state.graph

    # Ensure session_id is stored in session state
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = generate_session_id()

    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    col1, col2 = st.columns([0.7, 0.3], gap="small", vertical_alignment="bottom")
    with col1:
        st.title("TFG Tutor Chatbot")
    with col2:
        # Add blank lines to push the button down
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px;">
                <p style="font-weight: bold; margin: 0;">Session ID: {st.session_state['session_id']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Refresh Chat Session", type="primary"):
            st.session_state["session_id"] = generate_session_id()
            st.session_state["chat_history"] = []
            st.rerun()

    # Build pipeline_messages list from the chat history
    pipeline_messages = []
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            pipeline_messages.append(HumanMessage(chat["message"]))
        elif chat["role"] == "assistant":
            pipeline_messages.append(AIMessage(chat["message"]))

    # Display previous chat messages
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])

    # Chatbot interface using st.chat_input and st.chat_message
    user_input = st.chat_input("Ask your question about TFGs or publications")
    if user_input:
        # Append user's message to chat history and pipeline messages
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        pipeline_messages.append(HumanMessage(user_input))

        # Process the user's question through the retrieval/generation pipeline
        config = {
            "callbacks": [langfuse_handler],
            "run_name": app_name,
            "metadata": {
                "langfuse_session_id": st.session_state["session_id"]
            }
        }

        # Variable to accumulate the complete assistant response
        complete_response = ""

        def generate_streaming_message():
            nonlocal complete_response
            # Pass the full conversation history into the pipeline
            for message, metadata in graph.stream(
                {"messages": pipeline_messages},
                stream_mode="messages",
                config=config
            ):
                complete_response += message.content
                yield message.content

        with st.chat_message("assistant"):
            st.write_stream(generate_streaming_message())

        # Append the complete assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "message": complete_response})

if __name__ == "__main__":
    main()