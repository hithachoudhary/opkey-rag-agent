#Saves vectors straight into your persistent storage layers

from app.db.chroma import chroma_client

class EmbeddingService:
    def index_chunks(self, chunks: list) -> int:
        if not chunks:
            return 0
        ids = [f"id_{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks]
        documents = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        
        chroma_client.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

embedding_service = EmbeddingService()