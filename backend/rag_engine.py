from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Load local documents
documents = SimpleDirectoryReader("../documents").load_data()

# Use local LLM and embedding
llm = Ollama(model="llama3")
embed_model = OllamaEmbedding(model_name="llama3")

Settings.llm = llm
Settings.embed_model = embed_model

# Build the vector index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def query_rag(question: str) -> str:
    return query_engine.query(question).response
