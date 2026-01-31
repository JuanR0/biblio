from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str  # "general", "books", "computers", "cubicles"