# https://python.langchain.com/docs/tutorials/rag/
import streamlit as st
import logging
import os
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from utils import * # Import the utils.py functions
from langchain_core.documents import Document
from langchain_core.prompts.base import format_document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
import logging
# Own setup of LLM and Embeddings
from custom_chat_model import CustomLLM
from interference_api_embeddings import InferenceAPIEmbeddings

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

template = """You are a helpful assistant that answers questions related to TFGs (Final Degree Projects that Catalan students must do do end their degree) and scientific publications of the CVC (Computer Vision Center). 
    The CVC is a non-profit research center established in 1995 by the Generalitat de Catalunya and the Universitat Autònoma de Barcelona (UAB), focused on computer vision research, that closely cooperateswith the university spetially with these TFGs.

    Your goal is to answer user questions based on retrieved documents (TFGs or publications), using their **abstracts** and metadata fields. If no relevant documents are found, clearly inform the user.
    
    The user asked:
    
    {question}

    When documents are found and if relevant, structure the response as follows:

    For **publications**:
    *Title*: (title of the publication)
        - **Authors**: (list of authors) -> turn the list into a string with "," as separator
        - **Abstract**: (SUMMARIZED abstract)
        - **Published in**: (year)
        - **Link**: (hyperlink to publication) (if None do not include it as listed)
        
    For **TFGs**:
    *Title*: (title of tfg)
        - **Authors**: (list of authors) -> turn the list into a string with "," as separator
        - **Abstract**: (SUMMARIZED abstract)
        - **Delivered in**: (year)
        - **Link**: (hyperlink to TFG) (if None do not include it as listed)
    
    Please if any NULL/EMPTY field do NOT include in in the output. After listing the results create a brief explanation of the retrieved documents at the end of the response.
    Even having around 10 results, please BY DEFAULT just show the TOP 5 most relevant ones based on your criteria. If not relevant for the query do not show them. If users specifically ask for more, then show some more results.
    Be direct and concise in your responses, focusing on the most relevant information.

    After listing the results:
    - Add a brief comparison of the documents where possible.
    - Highlight with **bold** the most relevant information and titles using *word*.
    - Focus your response on content related to the user's query, emphasizing **abstract-based relevance**.
    
    Please if user's query do not need to list any document, just answer the question directly!

    The retrieved documents are:
    {context}
    
    Answer the user's question:
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
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    #docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    docs_content = "\n\n".join(format_document(doc, document_prompt) for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

memory = MemorySaver()
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile(checkpointer=memory)

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["message"])

# Chatbot interface using st.chat_input and st.chat_message
user_input = st.chat_input("Ask your question about TFGs or publications")
if user_input:
    # Append user's message to chat history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    
    # Process the user's question through the retrieval/generation pipeline
    config = {"configurable": {"thread_id": "abc123"}}
    result = graph.invoke({"question": user_input}, config=config)
    assistant_answer = result["answer"]
    
    # Append assistant's answer to chat history
    st.session_state.chat_history.append({"role": "assistant", "message": assistant_answer})
    
    # Optionally, display the new messages immediately
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write(assistant_answer)
