from fastapi import FastAPI
import fitz
from app.services.pdf_service import pdf_service
from app.services.chunking_service import chunking_service

app = FastAPI(title="Opkey RAG Agent API", version="2.0.0")

@app.get("/test-chunks")
def test_chunks():
    try:
        file_path = "/workspace/data/oracle_financials_implementation_guide.pdf"
        doc = fitz.open(file_path)
        
        # Target Page 7 directly (the Introduction / Feature List Chapter)
        # This page contains excellent vertical lists to evaluate our structural changes
        target_page_num = 7
        raw_page_text = doc[target_page_num - 1].get_text()
        
        # Execute the new engineering chunker
        processed_chunks = chunking_service.split_text_into_chunks(
            raw_text=raw_page_text,
            filename="oracle_financials_implementation_guide.pdf",
            page_number=target_page_num,
            chunk_size=750 # Tighter constraint to witness sentence-overlap boundaries easily
        )
        
        return {
            "status": "success",
            "target_page_evaluated": target_page_num,
            "chunks_generated": len(processed_chunks),
            "sample_chunks": processed_chunks
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}