import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from app.services.pdf_service import pdf_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service
from app.db.chroma import chroma_client
from app.core.config import settings

# Setup unified terminal logging tracking format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("opkey_agent_api")

# Initialize the FastAPI App with formal configuration metadata
app = FastAPI(
    title="Opkey Enterprise RAG Agent API",
    version="2.0.0",
    description="Production-grade Retrieval-Augmented Generation engine for enterprise document analysis."
)

class QueryRequest(BaseModel):
    question: str


@app.get("/health", tags=["System Diagnostics"])
def health_check():
    """
    Evaluates system operational readiness and active resource state.
    """
    try:
        current_chunks = chroma_client.get_count()
        return {
            "status": "healthy",
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm": settings.LLM_MODEL,
            "indexed_chunks": current_chunks,
            "collection": settings.COLLECTION_NAME
        }
    except Exception as e:
        logger.error(f"Health checkpoint failure: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/ingest", tags=["Document Ingestion Pipeline"])
def ingest_document(force_rebuild: bool = False):
    """
    Data Engineering Pipeline:
    1. Extracts raw text from the enterprise PDF file repository.
    2. Filters structural layout noise, tables of contents, and boilerplate data.
    3. Segments raw text strings into sentence-bounded overlapping semantic windows.
    4. Automatically generates embeddings and updates the persistent vector storage index.
    """
    start_time = time.perf_counter()
    try:
        file_path = "/workspace/data/oracle_financials_implementation_guide.pdf"
        filename = "oracle_financials_implementation_guide.pdf"
        
        filtered_pages = pdf_service.extract_text_by_page(file_path)
        
        all_document_chunks = []
        for page in filtered_pages:
            chunks = chunking_service.split_text_into_chunks(
                raw_text=page["text"],
                filename=filename,
                page_number=page["page_number"]
            )
            all_document_chunks.extend(chunks)
            
        ingest_result = embedding_service.index_chunks(
            chunks=all_document_chunks, 
            force_rebuild=force_rebuild
        )
        
        execution_duration = round((time.perf_counter() - start_time) * 1000)
        
        return {
            "status": "processed",
            "response_time_ms": execution_duration,
            "pipeline_metrics": {
                "total_pages_retained": len(filtered_pages),
                "total_chunks_generated": len(all_document_chunks)
            },
            "database_ingestion_layer": ingest_result
        }
    except Exception as e:
        logger.error(f"Pipeline failure during ingestion: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/query", tags=["Conversational RAG Execution"])
def query_agent(payload: QueryRequest):
    """
    Orchestrated RAG Execution Loop with Context Threshold Guards.
    """
    start_time = time.perf_counter()
    try:
        logger.info(f"Processing context-grounded agent query: {payload.question}")
        
        # 1. Fetch relevant factual contexts from the vector storage system
        context_data = retrieval_service.retrieve_context(query=payload.question, top_k=5)
        retrieval_stats = context_data["summary"]
        
        confidence_label = retrieval_stats.pop("retrieval_confidence")
        
        # 2. Threshold Guard check to stop weak context processing immediately
        if retrieval_stats["highest_similarity"] < retrieval_stats["threshold"]:
            execution_duration = round((time.perf_counter() - start_time) * 1000)
            logger.info("Context Guard Triggered: Insufficient similarity proximity.")
            return {
                "status": "insufficient_context",
                "response_time_ms": execution_duration,
                "answer": "The provided Oracle documentation does not contain sufficient information to answer this question.",
                "confidence": {"retrieval": "Low"},
                "retrieval": retrieval_stats,
                "citations": [],
                "metrics": {
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "llm": settings.LLM_MODEL,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        
        # 3. Generate data-grounded contextual response block
        result = llm_service.generate_answer(
            question=payload.question, 
            context_data=context_data
        )
        
        # Append referenced citation footer explicitly to response string
        page_list_str = ", ".join(map(str, retrieval_stats["source_pages"]))
        final_answer = f"{result['generated_answer']}\n\n---\n**Referenced Documentation Pages:** {page_list_str}"
        
        execution_duration = round((time.perf_counter() - start_time) * 1000)
        logger.info(f"Query sequence resolved in {execution_duration} ms.")
        
        telemetry_metrics = result["token_metrics"]
        telemetry_metrics["embedding_model"] = settings.EMBEDDING_MODEL
        telemetry_metrics["llm"] = settings.LLM_MODEL
        
        return {
            "status": "success",
            "response_time_ms": execution_duration,
            "answer": final_answer,
            "confidence": {
                "retrieval": confidence_label
            },
            "retrieval": retrieval_stats,
            "citations": context_data["source_chunks"],
            "metrics": telemetry_metrics
        }
    except Exception as e:
        logger.error(f"Execution tracking exception during query lifecycle: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/debug/chunks/sample", tags=["System Diagnostics"])
def debug_chunk_segmentation():
    """
    Comprehensive Diagnostic Extraction Summary Route.
    """
    try:
        import fitz
        file_path = "/workspace/data/oracle_financials_implementation_guide.pdf"
        doc = fitz.open(file_path)
        
        target_page_num = 7
        raw_page_text = doc[target_page_num - 1].get_text()
        
        processed_chunks = chunking_service.split_text_into_chunks(
            raw_text=raw_page_text,
            filename="oracle_financials_implementation_guide.pdf",
            page_number=target_page_num
        )
        
        debug_output = []
        for c in processed_chunks:
            debug_output.append({
                "page": c["metadata"]["page_number"],
                "chunk_id": c["metadata"]["chunk_id"],
                "characters": c["metadata"]["character_metrics"],
                "estimated_tokens": c["metadata"]["estimated_tokens"],
                "chunk_preview": c["text"][:150] + "..."
            })
            
        return {
            "status": "success",
            "target_page_evaluated": target_page_num,
            "total_chunks_discovered": len(debug_output),
            "diagnostics": debug_output
        }
    except Exception as e:
        logger.error(f"Diagnostic extraction routine failure: {str(e)}")
        return {"status": "error", "message": str(e)}