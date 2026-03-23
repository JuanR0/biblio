from pydantic import BaseModel
from typing import Dict, List, Optional

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None 

class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str
    entities: Dict[str, List[str]]
    session_id: Optional[str] = None 