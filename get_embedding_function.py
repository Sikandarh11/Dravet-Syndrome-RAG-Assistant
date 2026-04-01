# get_embedding_function.py
# Returns an embedding "object" compatible with LangChain/FAISS.
# Uses the Ollama embeddings wrapper if available, else provides a minimal wrapper.

from typing import List
from langchain_core.embeddings import Embeddings
from config_loader import load_config

try:
    # Preferred: langchain_ollama provides OllamaEmbeddings compatible with LangChain
    from langchain_ollama import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None


class OllamaEmbeddingWrapper(Embeddings):
    """
    Minimal wrapper that extends Embeddings base class.
    It delegates to OllamaEmbeddings if available, otherwise it will attempt HTTP fallback.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        if OllamaEmbeddings is not None:
            try:
                self.client = OllamaEmbeddings(model=model)
            except Exception:
                self.client = None
        else:
            self.client = None

    def _request_embeddings(self, payload):
        import requests

        last_error = None
        # Support both newer and older Ollama embedding routes.
        for endpoint in ("/api/embed", "/api/embeddings"):
            try:
                resp = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=30)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_error = e
                continue

        if last_error is not None:
            raise RuntimeError(f"Failed to fetch embeddings from Ollama at {self.base_url}") from last_error
        raise RuntimeError(f"Embedding endpoint not found on {self.base_url}. Tried /api/embed and /api/embeddings")

    def _extract_embedding(self, data):
        embeddings = data.get("embeddings", [])
        if embeddings and isinstance(embeddings, list):
            first = embeddings[0]
            if isinstance(first, list):
                return first

        single = data.get("embedding", [])
        if single and isinstance(single, list):
            return single
        return []

    def _embed_one(self, text: str) -> List[float]:
        # Try modern payload first, then legacy fallback.
        payloads = [
            {"model": self.model, "input": text},
            {"model": self.model, "prompt": text},
        ]
        last_error = None
        for payload in payloads:
            try:
                data = self._request_embeddings(payload)
                vector = self._extract_embedding(data)
                if vector:
                    return vector
            except Exception as e:
                last_error = e
                continue

        if last_error is not None:
            raise RuntimeError("Ollama returned no embedding vector for the input text") from last_error
        raise RuntimeError("Ollama returned no embedding vector for the input text")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Newer Ollama builds support batch inputs on /api/embed.
        payloads = [
            {"model": self.model, "input": texts},
            {"model": self.model, "prompt": texts},
        ]

        for payload in payloads:
            try:
                data = self._request_embeddings(payload)
                vectors = data.get("embeddings", [])
                if vectors and isinstance(vectors, list) and len(vectors) == len(texts):
                    return vectors
            except Exception:
                continue

        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of strings. LangChain/FAISS expects a list-of-list floats.
        """
        if self.client:
            # many wrappers expose embed_documents; try direct call
            if hasattr(self.client, "embed_documents"):
                vectors = self.client.embed_documents(texts)
                if vectors:
                    return vectors
            # fallback to embed_query per item
            return [self.client.embed_query(t) for t in texts]

        if not texts:
            return []

        # HTTP fallback: use fast batch first, then maximum-compatibility per-item mode.
        vectors = self._embed_batch(texts)
        if vectors:
            return vectors
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        """
        if self.client:
            if hasattr(self.client, "embed_query"):
                return self.client.embed_query(text)
            if hasattr(self.client, "embed_documents"):
                return self.client.embed_documents([text])[0]

        # HTTP fallback
        return self._embed_one(text)


def get_embedding_function(model=None, base_url=None):
    """Return an embeddings object compatible with LangChain and FAISS.from_documents"""
    config = load_config()
    ollama_config = config.get("ollama", {})
    resolved_model = model or ollama_config.get("embedding_model", "nomic-embed-text")
    resolved_base_url = base_url or ollama_config.get("base_url", "http://localhost:11434")
    return OllamaEmbeddingWrapper(model=resolved_model, base_url=resolved_base_url)