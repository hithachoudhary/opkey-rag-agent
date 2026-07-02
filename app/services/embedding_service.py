import logging
from app.db.chroma import chroma_client
from app.core.config import settings

logger = logging.getLogger("opkey_agent_api")

class EmbeddingService:
    def index_chunks(self, chunks: list, force_rebuild: bool = False) -> dict:
        """
        Coordinates database ingestion with execution validation and advanced logging.
        """
        if not chunks:
            logger.info("ℹ️ No chunks provided for ingestion processing.")
            return {"status": "empty", "chunks_indexed": 0}
            
        # 1. Runtime API Key Guard (Allows server boot but blocks execution)
        if not settings.OPENAI_API_KEY:
            logger.error("Ingestion halted: OPENAI_API_KEY env variable is completely empty.")
            raise ValueError("Authentication Failed: Missing valid OPENAI_API_KEY in active runtime.")

        existing_count = chroma_client.get_count()
        
        # 2. Cost Guard Verification Layer
        if existing_count == len(chunks) and not force_rebuild:
            logger.info(f"[Cost Guard] Vector collection already holds {existing_count} chunks matching current payload. Skipping API execution.")
            return {
                "status": "already_indexed",
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunks_indexed": existing_count,
                "collection": settings.COLLECTION_NAME,
                "note": "Optimized workflow matched cache state. Zero OpenAI credits consumed."
            }
            
        # Parse arrays out for ingestion execution
        ids = [c['metadata']['chunk_id'] for c in chunks]
        documents = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        
        try:
            if force_rebuild or existing_count > 0:
                logger.warning("[force_rebuild] Resetting vector collection entries for a fresh index build...")
                chroma_client.clear_collection_records()
            
            logger.info(f"[OpenAI API] Generating embeddings using model: {settings.EMBEDDING_MODEL}")
            logger.info(f"Uploading {len(chunks)} text vectors into collection: {settings.COLLECTION_NAME}...")
            
            chroma_client.add_chunks(ids=ids, documents=documents, metadatas=metadatas)
            
            final_count = chroma_client.get_count()
            logger.info(f"Successfully indexed {final_count} chunks into ChromaDB.")
            
            return {
                "status": "success", 
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunks_indexed": final_count,
                "collection": settings.COLLECTION_NAME,
                "note": "New embeddings generated and indexed successfully via OpenAI."
            }
            
        except Exception as e:
            logger.error(f"Ingestion Processing Error: {str(e)}")
            raise e

embedding_service = EmbeddingService()