# https://python.langchain.com/docs/tutorials/rag/
import streamlit as st
import logging
import os
#from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
#from qdrant_client import QdrantClient
from utils import * # Import the utils.py functions
from langchain_core.documents import Document
from langchain_core.prompts.base import format_document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
# Own setup of LLM and Embeddings
#from custom_chat_model import CustomLLM
from interference_api_embeddings import InferenceAPIEmbeddings

def initialize_app(memory):
    st.set_page_config(
        page_title="RAG",
        page_icon=":desktop_computer:",
        layout="centered",
        initial_sidebar_state="collapsed", # To be changed if having filters!
    )
    st.title("TFG Tutor Chatbot")

    # --- Enforce Login ---
    _ ='''if not st.session_state.get("logged_in", False):
        st.switch_page("Welcome.py")
        st.stop()'''

    # --- App Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    check_connections()
    #llm = CustomLLM()
    llm = init_chat_model(
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_provider = "openai",
        api_key = "empty",
        base_url = "http://host.docker.internal:8111/v1", # "http://localhost:8111/v1",
        temperature = 0.7,
        max_tokens = 700,
    )
    embeddings = InferenceAPIEmbeddings()

    QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "publications")

    # Initialize vector store with Weaviate and custom embeddings
    # https://python.langchain.com/docs/integrations/vectorstores/qdrant/
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        content_payload_key = 'page_content',
        metadata_payload_key = 'metadata'
    )

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
        context: List[Document]
        messages: List[BaseMessage]

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["messages"][0].content)
        return {"context": retrieved_docs}

    def generate(state: State):
        messages = state["messages"]
        print(messages)
        question = messages[0].content
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
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph

memory = MemorySaver()
# Initialize the app (this part runs once per script execution)
graph = initialize_app(memory)

def main():
    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
        config = {"configurable": {"thread_id": "1"}}

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
