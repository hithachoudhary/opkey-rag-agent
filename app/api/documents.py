from fastapi import APIRouter, HTTPException
from app.db.chroma import chroma_client

router = APIRouter()

@router.get("/documents", tags=["Metadata Operations"])
def list_documents():
    data = chroma_client.collection.get()
    unique_docs = list({m['source'] for m in data['metadatas'] if 'source' in m}) if data['metadatas'] else []
    return {"documents": unique_docs}

@router.delete("/documents/{id}", tags=["Metadata Operations"])
def clear_vector_store():
    data = chroma_client.collection.get()
    if data['ids']:
        chroma_client.collection.delete(ids=data['ids'])
    return {"deleted": True}