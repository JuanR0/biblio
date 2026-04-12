from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pregunta de usuario")
    session_id: str = Field(..., description="Identificador de sesion")

class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str
    entities: Dict[str, List[str]]
    session_id: str  