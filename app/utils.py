from neo4j import GraphDatabase
import requests
import os
import streamlit as st
from langchain.chat_models import init_chat_model
from interference_api_embeddings import InferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore
import logging
import time
from langchain_core.documents import Document
from langchain_core.prompts.base import format_document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# Initialize the LLM
def init_llm():
    return init_chat_model(
        model=os.getenv("LLM_MODEL"),
        model_provider="openai",
        api_key="empty",
        base_url=os.getenv("LLM_HOST"),
        temperature=os.getenv("LLM_TEMPERATURE"),
        max_tokens=os.getenv("LLM_MAX_TOKENS"),
    )

def init_embeddings():
    return InferenceAPIEmbeddings()
    
def neo4j_get_session():
    AUTH = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    neo4j_driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=AUTH)
    neo4j_session = neo4j_driver.session(database="neo4j")
    return neo4j_session

def get_qdrant_vector_store(embeddings):
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT")
    qdrant_collection = os.getenv("QDRANT_COLLECTION")
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=qdrant_collection,
        url=f"http://{qdrant_host}:{qdrant_port}",
        content_payload_key = 'page_content',
        metadata_payload_key = 'metadata'
    )
    
def generate_session_id():
    # Generates if based on timestamp
    return str(int(time.time()))

# Utility function to check the status of a Qdrant collection using requests
def check_qdrant():
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = os.getenv("QDRANT_PORT")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an error for unsuccessful status codes
        col_info = response.json()
        print(col_info)
        return col_info["result"]["status"] == "green"
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")

# Utility function to check FastAPI health
def check_fastapi_health(type = "llm"):
    if type == "llm":
        FASTAPI_HOST = os.getenv("LLM_HOST")
        FASTAPI_PORT = os.getenv("LLM_PORT")
    if type == "embeddings":
        FASTAPI_HOST = os.getenv("EMBEDDINGS_HOST")
        FASTAPI_PORT = os.getenv("EMBEDDINGS_PORT")
    url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/health"
    
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        raise ConnectionError(f"Failed to connect to FastAPI: {str(e)}")

def check_connections():
    try:
        if not check_qdrant():
            st.error("Qdrant is not healthy. Please check the server.")
            return
    except ConnectionError as e:
        st.error(str(e))
        return

    '''try:
        if not check_fastapi_health(type="llm"):
            st.error("LLM FastAPI is not healthy. Please check the server.")
            return
    except ConnectionError as e:
        st.error(str(e))
        return'''

    try:
        if not check_fastapi_health(type="embeddings"):
            st.error("Embeddings FastAPI is not healthy. Please check the server.")
            return
    except ConnectionError as e:
        st.error(str(e))
        return

def format_semantic_entities(semantic_entities: dict, candidates: int) -> str:
    """
    Returns a human-readable string of relevant keywords and authors
    from the semantic_entities dictionary, ignoring scores.
    Only the top `candidates` items from each category will be included.
    """
    if not semantic_entities:
        return ""

    # Slice the lists to include only the top `candidates`
    keywords = [item["candidate"] for item in semantic_entities.get("Keyword", [])][:candidates]
    people = [item["candidate"] for item in semantic_entities.get("Person", [])][:candidates]

    output_lines = []

    if keywords:
        output_lines.append("Relevant Keywords for the user query:")
        output_lines.append(", ".join(keywords))
        output_lines.append("")  # blank line

    if people:
        output_lines.append("Relevant Authors for the user query:")
        output_lines.append(", ".join(people))
        output_lines.append("")  # blank line

    return "\n".join(output_lines).strip()

def get_rag_graph(llm, vector_store):
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