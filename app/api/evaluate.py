from fastapi import APIRouter
from app.services.evaluation_service import evaluation_service

router = APIRouter()

@router.get("/evaluate", tags=["Evaluation Suite"])
def evaluate_agent():
    return evaluation_service.evaluate_pipeline()