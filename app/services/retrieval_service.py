#Performs semantic context mapping over the entire collection

from app.db.chroma import chroma_client

class RetrievalService:
    def get_relevant_context(self, query: str, top_k: int = 5) -> list:
        results = chroma_client.collection.query(query_texts=[query], n_results=top_k)
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })
        return formatted

retrieval_service = RetrievalService()