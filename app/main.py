import os
import time
import json
import logging
from typing import List
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Pipeline Layer Service Components Imports
from app.services.pdf_service import pdf_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service
from app.db.chroma import chroma_client
from app.core.config import settings

# Ensure secure operational scratch space exists for ingestion streams
os.makedirs("/workspace/data", exist_ok=True)

# Initialize unified application execution logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("opkey_agent_api")

app = FastAPI(
    title="Opkey Enterprise RAG Agent API",
    version="2.0.0",
    description="Production-grade document processing and context-grounded semantic analysis engine."
)

class QueryRequest(BaseModel):
    question: str


# 1. SYSTEM DIAGNOSTICS & TELEMETRY ENDPOINTS

@app.get("/health", tags=["System Diagnostics"])
def health_check():
    """
    Evaluates system operational readiness and current vector index statistics.
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
        logger.error(f"Health checkpoint telemetry failure: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/evaluate", tags=["System Diagnostics"])
def get_evaluation_metrics(mode: str = Query("basic", enum=["basic", "judge"], description="Select target evaluation view report profile rules.")):
    """
    Exposes two complementary evaluation modes over HTTP:
    - basic: measures system-level retrieval, token counts, and processing performance.
    - judge: measures semantic output quality (Hit Rate, Faithfulness, Relevance) graded by an LLM judge.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Route target path mapping directly to the new isolated folder tracking files
    report_path = os.path.join(BASE_DIR, "tests", "reports", f"{mode}_report.json")
    
    if not os.path.exists(report_path):
        return {
            "status": "pending",
            "message": f"The requested {mode}_report.json is not available yet. Please run the evaluation script with '--mode {mode}' in the container terminal."
        }
        
    try:
        with open(report_path, "r") as f:
            report_data = json.load(f)
            
        return {
            "status": "success",
            "test_suite": report_data.get("test_suite"),
            "evaluation_mode": report_data.get("evaluation_mode"),
            "metrics": report_data.get("metrics"),
            "detailed_runs": report_data.get("detailed_runs", [])
        }
    except Exception as e:
        logger.error(f"Failed to read evaluation asset payload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 2. DOCUMENT LIFECYCLE MANAGEMENT ENDPOINTS

@app.post("/ingest", tags=["Document Ingestion Pipeline"])
async def ingest_document(
    file: UploadFile = File(...), 
    force_rebuild: bool = Query(False, description="Clears existing collection space prior to execution.")
):
    """
    Dynamically processes incoming document buffers into vector storage space.
    """
    start_time = time.perf_counter()
    temp_file_path = f"/workspace/data/tmp_{file.filename}"
    
    try:
        # Stream incoming payload out to disk memory architecture
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
            
        # Extract clean, page-bounded string data from PDF structure
        filtered_pages = pdf_service.extract_text_by_page(temp_file_path)
        
        all_document_chunks = []
        for page in filtered_pages:
            chunks = chunking_service.split_text_into_chunks(
                raw_text=page["text"],
                filename=file.filename,
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
            "filename": file.filename,
            "pipeline_metrics": {
                "total_pages_retained": len(filtered_pages),
                "total_chunks_generated": len(all_document_chunks)
            },
            "database_ingestion_layer": ingest_result
        }
    except Exception as e:
        logger.error(f"Pipeline dynamic ingestion failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failure: {str(e)}")
    finally:
        # Sweeper step to verify scratch memory bounds are kept clear
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/documents", tags=["Document Lifecycle Management"])
def list_documents():
    """
    Inspects active vector data components to extract a deduplicated registry map of indexed files.
    """
    try:
        collection_data = chroma_client.collection.get(include=["metadatas"])
        metadatas = collection_data.get("metadatas", [])
        
        if not metadatas:
            return {"status": "success", "documents": []}
            
        doc_registry = {}
        for meta in metadatas:
            doc_name = meta.get("filename") or meta.get("source") or "unknown_document"
            page_num = meta.get("page_number", 0)
            
            if doc_name not in doc_registry:
                doc_registry[doc_name] = {
                    "doc_name": doc_name,
                    "chunks": 0,
                    "pages_set": set()
                }
                
            doc_registry[doc_name]["chunks"] += 1
            if page_num > 0:
                doc_registry[doc_name]["pages_set"].add(page_num)
                
        formatted_documents = []
        for name, profile in doc_registry.items():
            formatted_documents.append({
                "doc_name": name,
                "pages": len(profile["pages_set"]),
                "chunks": profile["chunks"]
            })
            
        return {"status": "success", "documents": formatted_documents}
    except Exception as e:
        logger.error(f"Failed to compile document registry profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_name:path}", tags=["Document Lifecycle Management"])
def delete_document(document_name: str):
    """
    Isolates and scrubs all vector fragments matching the targeted document string identifier.
    """
    try:
        logger.info(f"Initiating isolated record deletion sequence for: {document_name}")
        
        collection_data = chroma_client.collection.get(include=["metadatas"])
        metadatas = collection_data.get("metadatas", [])
        
        exists = any((m.get("filename") == document_name or m.get("source") == document_name) for m in metadatas)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found in target store.")
            
        # Deletes items matching either metadata schema variant for structural safety
        chroma_client.collection.delete(where={"filename": document_name})
        chroma_client.collection.delete(where={"source": document_name})
        
        logger.info(f"Successfully deleted all vector blocks belonging to: {document_name}")
        return {"status": "success", "deleted": True, "document_name": document_name}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Exception encountered during delete collection sweep: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# 3. SEMANTIC QUERY PROCESSING ENGINE
# =========================================================================

@app.post("/query", tags=["Conversational RAG Execution"])
def query_agent(payload: QueryRequest):
    """
    Context-grounded production inference query loop with similarity metrics extraction.
    """
    start_time = time.perf_counter()
    try:
        logger.info(f"Processing agent execution path for statement: {payload.question}")
        
        # 1. Query vector architecture for candidate reference chunks
        context_data = retrieval_service.retrieve_context(query=payload.question, top_k=5)
        retrieval_stats = context_data["summary"]
        
        confidence_label = retrieval_stats.pop("retrieval_confidence")
        
        # 2. Context Guard Threshold Check
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
        
        # 3. Generate structured contextual response block
        result = llm_service.generate_answer(
            question=payload.question, 
            context_data=context_data
        )
        
        # Construct clean page-reference documentation string block footer
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


# =========================================================================
# 4. DIAGNOSTIC DEBUG ENDPOINTS
# =========================================================================

@app.get("/debug/chunks/sample", tags=["System Diagnostics"])
def debug_chunk_segmentation():
    """
    Comprehensive Diagnostic Extraction Summary Route.
    """
    try:
        import fitz
        file_path = "/workspace/data/oracle_financials_implementation_guide.pdf"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Reference verification PDF asset missing from workspace directory mount.")
            
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
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Diagnostic extraction routine failure: {str(e)}")
        return {"status": "error", "message": str(e)}