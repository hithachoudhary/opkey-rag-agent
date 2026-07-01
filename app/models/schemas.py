from pydantic import BaseModel, Field
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., example="What primary functional areas are covered within the Financials offering?")
    top_k: int = Field(default=5, ge=1, le=10)

class ChunkSource(BaseModel):
    text: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkSource]