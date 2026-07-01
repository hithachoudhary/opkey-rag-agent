from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service

router = APIRouter()

@router.post("/query", response_model=QueryResponse, tags=["RAG Core"])
async def query_agent(payload: QueryRequest):
    try:
        contexts = retrieval_service.get_relevant_context(payload.question, top_k=payload.top_k)
        answer = llm_service.synthesize_answer(payload.question, contexts)
        return {"answer": answer, "sources": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))