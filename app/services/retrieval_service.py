from app.db.chroma import chroma_client

class RetrievalService:
    def retrieve_context(self, query: str, top_k: int = 3) -> dict:
        """
        Queries the vector database for the most contextually relevant chunks
        and formats them into a clean payload for the LLM.
        """
        try:
            # 1. Fetch matches from ChromaDB client wrapper
            results = chroma_client.query_similarity(query_text=query, n_results=top_k)
            
            # Since the database is empty right now, handle empty fetches gracefully
            if not results or not results.get("documents") or not results["documents"][0]:
                return {
                    "context_string": "No relevant context found (Database currently unpopulated).",
                    "source_chunks": []
                }
            
            # 2. Extract texts and combine them with clear segment separators
            retrieved_texts = results["documents"][0]
            retrieved_metadatas = results["metadatas"][0]
            
            context_string = "\n\n---\n\n".join(retrieved_texts)
            
            # 3. Compile clean structural source tracking references
            source_chunks = []
            for idx, meta in enumerate(retrieved_metadatas):
                source_chunks.append({
                    "chunk_id": meta.get("chunk_id"),
                    "page_number": meta.get("page_number"),
                    "text_preview": retrieved_texts[idx][:150] + "..."
                })
                
            return {
                "context_string": context_string,
                "source_chunks": source_chunks
            }
            
        except Exception as e:
            # Safe mock fallback for development mode before vector indexing is complete
            print(f"ℹ️ [RetrievalService] Querying unindexed db: {str(e)}. Returning structural placeholder.")
            return {
                "context_string": f"[Mock Context for Question: '{query}'] Oracle Fusion Cloud Financials ledger configuration involves mapping your chart of accounts, accounting calendar, and primary currency.",
                "source_chunks": [
                    {
                        "chunk_id": "oracle_p007_c000",
                        "page_number": 7,
                        "text_preview": "Mock context chunk tracking verified..."
                    }
                ]
            }

retrieval_service = RetrievalService()