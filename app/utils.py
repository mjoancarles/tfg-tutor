from neo4j import GraphDatabase
import requests
import os
import streamlit as st
from langchain.chat_models import init_chat_model
from interference_api_embeddings import InferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore
import logging
import time

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

def format_semantic_entities(semantic_entities: dict) -> str:
    if not semantic_entities:
        return ""
    formatted_parts = []
    for entity_type, results in semantic_entities.items():
        if results:
            candidates = ", ".join(
                [f"{item['candidate']} ({item['score']:.2f})" for item in results]
            )
            formatted_parts.append(f"{entity_type}: {candidates}")
    return "; ".join(formatted_parts)
