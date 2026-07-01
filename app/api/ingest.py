from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from app.services.pdf_service import pdf_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service

router = APIRouter()

@router.post("/ingest", tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=422, detail="Only PDFs are supported.")
    
    file_path = f"/workspace/data/{file.filename}"
    os.makedirs("/workspace/data", exist_ok=True)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pages_data = pdf_service.extract_text_by_page(file_path)
        chunks = chunking_service.split_pages_into_chunks(pages_data, file.filename)
        count = embedding_service.index_chunks(chunks)
        
        return {"filename": file.filename, "status": "successfully processed", "chunks_added": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))