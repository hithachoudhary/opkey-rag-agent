from fastapi import FastAPI
from pydantic import BaseModel
import fitz
from app.services.pdf_service import pdf_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service
from app.db.chroma import chroma_client

# Initialize the FastAPI App
app = FastAPI(title="Opkey RAG Agent API", version="2.0.0")

# Request validation schema for downstream execution
class QueryRequest(BaseModel):
    question: str


@app.post("/ingest-full-guide")
def ingest_full_guide():
    """
    Data Engineering Pipeline:
    1. Extracts raw text from the source PDF.
    2. Applies layout noise filters (wiping Cover pages, TOC, and legal sections).
    3. Slices text blocks using sentence-based recursive chunking.
    4. Registers chunks and metadata structural payloads into ChromaDB.
    """
    try:
        file_path = "/workspace/data/oracle_financials_implementation_guide.pdf"
        filename = "oracle_financials_implementation_guide.pdf"
        
        # 1. Read manual pages and apply layout noise filters
        filtered_pages = pdf_service.extract_text_by_page(file_path)
        
        # 2. Chunk text blocks using sentence boundary alignment rules
        all_document_chunks = []
        for page in filtered_pages:
            chunks = chunking_service.split_text_into_chunks(
                raw_text=page["text"],
                filename=filename,
                page_number=page["page_number"]
            )
            all_document_chunks.extend(chunks)
            
        # 3. Attempt database ingestion through our interface wrapper
        ingest_result = embedding_service.index_chunks(all_document_chunks)
        
        return {
            "status": "processed",
            "total_pages_retained": len(filtered_pages),
            "total_chunks_generated": len(all_document_chunks),
            "database_ingestion_layer": ingest_result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/query")
def query_agent(payload: QueryRequest):
    """
    RAG Orchestration Workflow:
    1. Accepts a natural language question.
    2. Retrieves contextually relevant chunks from ChromaDB (with dynamic catch fallbacks).
    3. Builds an isolation prompt binding system guards and manual facts.
    4. Prepares the generative response along with citations and metrics trackers.
    """
    try:
        # 1. Fetch relevant factual contexts from the vector storage system
        context_data = retrieval_service.retrieve_context(query=payload.question)
        
        # 2. Feed context and question to the LLM module to generate the validated response
        result = llm_service.generate_answer(
            question=payload.question, 
            context_data=context_data
        )
        
        return {
            "status": "success",
            "query_executed": payload.question,
            "response": result["generated_answer"],
            "citations": result["sources_used"],
            "metrics": {
                "estimated_prompt_tokens": result["tokens_estimated"]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/test-chunks")
def test_chunks():
    """
    Isolated Diagnostic Route:
    Target Page 7 directly to easily evaluate sentence-overlap boundaries, 
    bullet layout preservation, and regex noise performance.
    """
    try:
        file_path = "/workspace/data/oracle_financials_implementation_guide.pdf"
        doc = fitz.open(file_path)
        
        target_page_num = 7
        raw_page_text = doc[target_page_num - 1].get_text()
        
        processed_chunks = chunking_service.split_text_into_chunks(
            raw_text=raw_page_text,
            filename="oracle_financials_implementation_guide.pdf",
            page_number=target_page_num
        )
        
        return {
            "status": "success",
            "target_page_evaluated": target_page_num,
            "chunks_generated": len(processed_chunks),
            "sample_chunks": processed_chunks
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    