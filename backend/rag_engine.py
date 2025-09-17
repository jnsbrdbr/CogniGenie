# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, List, Optional

# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.core.memory import ChatMemoryBuffer
# from fastapi import FastAPI, Depends, HTTPException
# from pydantic import BaseModel
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from datetime import datetime, timedelta
# from pathlib import Path
# import os



# # Lazily-initialized globals
# _index: Optional[VectorStoreIndex] = None
# _query_engine = None



# # ADD: force consistent metadata keys we can rely on later
# def _file_meta(p: str):
#     pp = Path(p)
#     return {"file_path": str(pp), "file_name": pp.name}

# def _docs_dir() -> Path:
#     # Documents live at repo_root/documents; this file is repo_root/backend/rag_engine.py
#     return Path(__file__).resolve().parent.parent / "documents"


# def _build_index() -> None:
#     global _index, _query_engine

#     # Settings.llm = Ollama(model="llama3")
#     # Settings.embed_model = OllamaEmbedding(model_name="llama3")

#     docs_path = _docs_dir()
#     documents = SimpleDirectoryReader(
#         str(docs_path),
#         recursive=True,
#         filename_as_id=True,      # doc_id becomes the filename
#         file_metadata=_file_meta  # injects file_name + file_path
#     ).load_data()

#     splitter = SentenceSplitter(chunk_size=800, chunk_overlap=120)
#     nodes = splitter.get_nodes_from_documents(documents)

#     _index = VectorStoreIndex(nodes)
#     _query_engine = _index.as_query_engine(similarity_top_k=10)
    
#     Settings.llm = Ollama(
#     model="llama3",
#     base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
#     request_timeout=600.0,   # allow cold-start
#     keep_alive="2h",
# )
#     Settings.embed_model = OllamaEmbedding(
#         model_name="llama3",
#         base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
#         request_timeout=600.0,
# )
    



# def ensure_ready() -> None:
#     """Build the index on first use (or after reload)."""
#     global _query_engine
#     if _query_engine is None:
#         _build_index()


# def reload_index() -> None:
#     """Rebuild the index, reloading any new/changed documents."""
#     _build_index()


# def query_rag(question: str) -> str:
#     """Query the index and return just the answer text."""
#     ensure_ready()
#     resp = _query_engine.query(question)
#     return getattr(resp, "response", None) or str(resp)

# def query_rag_with_sources(question: str, top_k: int = 5):
#     ensure_ready()
#     resp = _query_engine.query(question)
#     sources = []
#     for sn in (resp.source_nodes or [])[:top_k]:
#         # Try common metadata keys that hold the file path/name
#         meta = getattr(sn, "metadata", None) or getattr(sn.node, "metadata", {}) or {}
#         fp = meta.get("file_path") or meta.get("filepath") or meta.get("filename") or meta.get("file_name")
#         fp = (
#             meta.get("file_name")   # prefer what we injected
#             or meta.get("file_path")
#             or meta.get("filepath")
#             or meta.get("filename")
#             or meta.get("source")
#             or meta.get("path")
#         )
#         doc_id = Path(fp).name if fp else (getattr(sn.node, "doc_id", None) or getattr(sn.node, "id_", "unknown"))
#         score = float(getattr(sn, "score", 0.0) or 0.0)
#         sources.append({"doc_id": doc_id, "score": score})
#     return {
#         "answer": getattr(resp, "response", None) or str(resp),
#         "sources": sources
#     }


# SECRET_KEY = "your-secret-key"
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 60

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Fake users DB
# fake_users = {
#     "employer": {"username": "employer", "password": pwd_context.hash("employer123"), "role": "employer"},
#     "employee": {"username": "employee", "password": pwd_context.hash("employee123"), "role": "employee"}
# }

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class UserLogin(BaseModel):
#     username: str
#     password: str

# class ChatRequest(BaseModel):
#     question: str

# app = FastAPI()

# def verify_password(plain, hashed):
#     return pwd_context.verify(plain, hashed)

# def create_access_token(data: dict, expires_delta: timedelta = None):
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# @app.post("/login", response_model=Token)
# def login(user: UserLogin):
#     db_user = fake_users.get(user.username)
#     if not db_user or not verify_password(user.password, db_user["password"]):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = create_access_token({"sub": user.username, "role": db_user["role"]})
#     return {"access_token": token, "token_type": "bearer"}

# def retrieve_sources(question: str, top_k: int = 5):
#     """Pure retrieval (no LLM). Returns [{'doc_id': <filename>, 'score': <float>}, ...]."""
#     ensure_ready()
#     retriever = _index.as_retriever(similarity_top_k=top_k)
#     nodes = retriever.retrieve(question)

#     out = []
#     for sn in nodes:
#         meta = getattr(sn, "metadata", None) or getattr(sn.node, "metadata", {}) or {}
#         fp = (
#             meta.get("file_name")
#             or meta.get("file_path")
#             or meta.get("filepath")
#             or meta.get("filename")
#             or meta.get("source")
#             or meta.get("path")
#         )
#         doc_id = Path(fp).name if fp else (getattr(sn.node, "doc_id", None) or getattr(sn.node, "id_", "unknown"))
#         score = float(getattr(sn, "score", 0.0) or 0.0)
#         out.append({"doc_id": doc_id, "score": score})
#     return out


# backend/rag_engine.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Ensure local Ollama calls don't get proxied; set default host ---
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

# Lazily-initialized globals
_index: Optional[VectorStoreIndex] = None
_query_engine = None


# Consistent metadata keys we can rely on later
def _file_meta(p: str) -> Dict[str, str]:
    pp = Path(p)
    return {"file_path": str(pp), "file_name": pp.name}


def _docs_dir() -> Path:
    # Documents live at repo_root/documents; this file is repo_root/backend/rag_engine.py
    return Path(__file__).resolve().parent.parent / "documents"


def _build_index() -> None:
    """(Re)build the index and query engine."""
    global _index, _query_engine

    # 1) Configure LLM + embeddings FIRST (explicit base_url + long timeouts)
    Settings.llm = Ollama(
        model="llama3",
        base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        request_timeout=600.0,  # survive cold starts
        keep_alive="2h",
    )
    Settings.embed_model = OllamaEmbedding(
        model_name="llama3",
        base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        request_timeout=600.0,
    )

    # 2) Load docs and build index
    docs_path = _docs_dir()
    documents = SimpleDirectoryReader(
        str(docs_path),
        recursive=True,
        filename_as_id=True,   # doc_id becomes the filename
        file_metadata=_file_meta
    ).load_data()

    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=120)
    nodes = splitter.get_nodes_from_documents(documents)

    _index = VectorStoreIndex(nodes)
    _query_engine = _index.as_query_engine(similarity_top_k=10)


def ensure_ready() -> None:
    """Build the index on first use (or after reload)."""
    global _query_engine
    if _query_engine is None:
        _build_index()


def reload_index() -> None:
    """Rebuild the index, reloading any new/changed documents."""
    _build_index()


def query_rag(question: str) -> str:
    """Query the index and return just the answer text."""
    ensure_ready()
    resp = _query_engine.query(question)
    return getattr(resp, "response", None) or str(resp)


def query_rag_with_sources(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Query and return answer + top-k source filenames with scores."""
    ensure_ready()
    resp = _query_engine.query(question)

    sources: List[Dict[str, Any]] = []
    source_nodes = getattr(resp, "source_nodes", None) or []
    for sn in source_nodes[:top_k]:
        # Try common metadata keys that hold the file path/name
        meta = getattr(sn, "metadata", None) or getattr(sn, "node", None)
        meta = getattr(meta, "metadata", {}) if hasattr(meta, "metadata") else (meta or {})

        fp = (
            meta.get("file_name")  # prefer what we injected
            or meta.get("file_path")
            or meta.get("filepath")
            or meta.get("filename")
            or meta.get("source")
            or meta.get("path")
        )
        # Fallbacks if no filepath-like metadata was found
        doc_id = Path(fp).name if fp else (
            getattr(getattr(sn, "node", None), "doc_id", None)
            or getattr(getattr(sn, "node", None), "id_", "unknown")
        )
        score = float(getattr(sn, "score", 0.0) or 0.0)
        sources.append({"doc_id": doc_id, "score": score})

    return {
        "answer": getattr(resp, "response", None) or str(resp),
        "sources": sources,
    }


def retrieve_sources(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Pure retrieval (no LLM). Returns:
    [{'doc_id': <filename>, 'score': <float>}, ...]
    """
    ensure_ready()
    retriever = _index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    out: List[Dict[str, Any]] = []
    for sn in nodes:
        meta = getattr(sn, "metadata", None) or getattr(sn, "node", None)
        meta = getattr(meta, "metadata", {}) if hasattr(meta, "metadata") else (meta or {})

        fp = (
            meta.get("file_name")
            or meta.get("file_path")
            or meta.get("filepath")
            or meta.get("filename")
            or meta.get("source")
            or meta.get("path")
        )
        doc_id = Path(fp).name if fp else (
            getattr(getattr(sn, "node", None), "doc_id", None)
            or getattr(getattr(sn, "node", None), "id_", "unknown")
        )
        score = float(getattr(sn, "score", 0.0) or 0.0)
        out.append({"doc_id": doc_id, "score": score})

    return out
