import logging
import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

# Initialize structured logging
logger = logging.getLogger("opkey_agent_api")

class ChromaDBClient:
    def __init__(self):
        # 1. Mount persistent storage strictly via configuration settings
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.openai_ef = None
        self.collection = None
        
        # 2. Resilient initial setup check
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.warning("[ChromaDB] OPENAI_API_KEY missing from environment. Vector endpoints will require validation on call.")
            # Set up a dummy function so the client object instantiates without crashing
            self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="PENDING_VALIDATION", 
                model_name=settings.EMBEDDING_MODEL
            )
        else:
            self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key, 
                model_name=settings.EMBEDDING_MODEL
            )
            
        # 3. Establish the target vector collection
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME, 
            embedding_function=self.openai_ef
        )

    def add_chunks(self, ids: list, documents: list, metadatas: list):
        """Inserts text blocks and structural metadata."""
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query_similarity(self, query_text: str, n_results: int = 3) -> dict:
        """Executes a semantic vector proximity search."""
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def clear_collection_records(self):
        """Wipes current collection records completely."""
        current_data = self.collection.get()
        if current_data and current_data["ids"]:
            self.collection.delete(ids=current_data["ids"])

    def get_count(self) -> int:
        """Returns total records currently stored inside the vector collection."""
        return self.collection.count()

chroma_client = ChromaDBClient()