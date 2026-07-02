import logging
from app.db.chroma import chroma_client

logger = logging.getLogger("opkey_agent_api")

class RetrievalService:
    def get_relevance_label(self, score: float) -> str:
        """Maps a mathematical similarity score to an intuitive presentation label."""
        if score >= 0.80:
            return "Very High"
        elif score >= 0.70:
            return "High"
        elif score >= 0.60:
            return "Medium"
        return "Low"

    def retrieve_context(self, query: str, top_k: int = 5) -> dict:
        """
        Queries ChromaDB for the top K most relevant text segments,
        calculating metrics optimized for downstream consumption.
        """
        try:
            logger.info(f"Querying vector store for: '{query}' with top_k={top_k}")
            results = chroma_client.collection.query(
                query_texts=[query], 
                n_results=top_k
            )
            
            if not results or not results.get("documents") or not results["documents"][0]:
                logger.warning("No relevant matching vectors discovered in the collection.")
                return {
                    "context_string": "",
                    "source_chunks": [],
                    "summary": {
                        "top_k": top_k,
                        "threshold": 0.60,
                        "highest_similarity": 0.0,
                        "lowest_similarity": 0.0,
                        "average_similarity": 0.0,
                        "retrieval_confidence": "Low",
                        "source_pages": [],
                        "retrieved_chunk_ids": []
                    }
                }
            
            retrieved_texts = results["documents"][0]
            retrieved_metadatas = results["metadatas"][0]
            retrieved_distances = results["distances"][0]
            
            context_string = "\n\n--- Document Segment Divider ---\n\n".join(retrieved_texts)
            
            citations = []
            scores = []
            source_pages = set()
            retrieved_chunk_ids = []
            
            for idx, meta in enumerate(retrieved_metadatas):
                raw_dist = retrieved_distances[idx]
                score = round(1.0 - (raw_dist / 2.0), 4)
                scores.append(score)
                
                page_num = int(meta.get("page_number", 0))
                source_pages.add(page_num)
                
                chunk_id = meta.get("chunk_id")
                retrieved_chunk_ids.append(chunk_id)
                
                citations.append({
                    "page": page_num,
                    "score": score,
                    "relevance": self.get_relevance_label(score),
                    "chunk_id": chunk_id,
                    "excerpt": retrieved_texts[idx].strip()
                })
                
            # Keep ranked strictly by descending similarity score
            citations = sorted(citations, key=lambda x: x["score"], reverse=True)
            
            highest = max(scores) if scores else 0.0
            lowest = min(scores) if scores else 0.0
            average = round(sum(scores) / len(scores), 3) if scores else 0.0
            
            return {
                "context_string": context_string,
                "source_chunks": citations,
                "summary": {
                    "top_k": top_k,
                    "threshold": 0.60,
                    "highest_similarity": highest,
                    "lowest_similarity": lowest,
                    "average_similarity": average,
                    "retrieval_confidence": self.get_relevance_label(highest),
                    "source_pages": sorted(list(source_pages)),
                    "retrieved_chunk_ids": retrieved_chunk_ids
                }
            }
            
        except Exception as e:
            logger.error(f"Error during context retrieval loop execution: {str(e)}")
            raise e

retrieval_service = RetrievalService()