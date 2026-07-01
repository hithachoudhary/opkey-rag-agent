import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

class ChromaDBClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY, 
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME, 
            embedding_function=self.openai_ef
        )

chroma_client = ChromaDBClient()