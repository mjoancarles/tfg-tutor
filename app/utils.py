import json
import requests
import os
import streamlit as st

# Utility function to check the status of a Qdrant collection using requests
def check_qdrant():
    QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "publications")
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
        FASTAPI_HOST = os.getenv("LLM_HOST", "host.docker.internal")
        FASTAPI_PORT = os.getenv("LLM_PORT", "8070")
    if type == "embeddings":
        FASTAPI_HOST = os.getenv("EMBEDDINGS_HOST", "host.docker.internal")
        FASTAPI_PORT = os.getenv("EMBEDDINGS_PORT", "8099")
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
