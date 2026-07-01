import os
import chromadb
from chromadb.utils import embedding_functions

class ChromaDBClient:
    def __init__(self):
        # 1. Mount persistent storage inside your Docker volume path
        self.db_path = "/workspace/chroma_db"
        self.collection_name = "oracle_financials_collection"
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 2. Setup the OpenAI Embedding Function (Place-holder for now)
        api_key = os.getenv("OPENAI_API_KEY", "MOCK_KEY_UNTIL_TOP_UP")
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        
        # 3. Establish the target vector collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.openai_ef
        )

    def add_chunks(self, ids: list, documents: list, metadatas: list):
        """Inserts text blocks and structural metadata dictionaries."""
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query_similarity(self, query_text: str, n_results: int = 3) -> dict:
        """Executes a semantic vector proximity search."""
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def clear_collection_records(self):
        """Wipes existing entries cleanly to prevent key collision loops on ingestion reload."""
        current_data = self.collection.get()
        if current_data and current_data["ids"]:
            self.collection.delete(ids=current_data["ids"])

    def get_count(self) -> int:
        """Returns total records currently stored inside the vector collection."""
        return self.collection.count()

chroma_client = ChromaDBClient()