import requests
from langchain.embeddings.base import Embeddings
from typing import List
import os

class InferenceAPIEmbeddings(Embeddings):
    
    def __init__(self):
        embeddings_host = os.getenv("EMBEDDINGS_HOST", "localhost")
        embeddings_port = os.getenv("EMBEDDINGS_PORT", "8099")
        self.api_url = f"http://{embeddings_host}:{embeddings_port}/generate_embeddings/"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {"sentences": texts}  # Updated to match FastAPI's expected payload structure
        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            return response.json()["embeddings"]  # Return the entire list of embeddings
        else:
            raise ValueError(f"Failed to get embeddings: {response.status_code} - {response.text}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]  # Embed a single query and return its embedding
