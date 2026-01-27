from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Modelos
class Question(BaseModel):
    question: str
    # user_id: Optional[str] = None

class Answer(BaseModel):
    answer: str
    confidence: float = 1.0
    entities: list = []
    suggested_questions: list = []

# Endpoint principal del asistente
@router.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """Procesa preguntas sobre el reglamento de biblioteca"""
    
    # Por ahora, lógica simulada
    user_question = question.question.lower()
    
    if "hola" in user_question:
        return Answer(
            answer="¡Hola! Soy el asistente de la biblioteca. ¿En qué puedo ayudarte?",
            confidence=0.9,
            entities=[{"text": "biblioteca", "label": "ORG"}],
            suggested_questions=[
                "¿Cuál es el horario?",
                "¿Cuántos libros puedo prestar?"
            ]
        )
    elif "horario" in user_question or "hora" in user_question:
        return Answer(
            answer="Abrimos de lunes a viernes de 8:00 a 20:00, sábados de 9:00 a 14:00",
            confidence=0.95
        )
    elif "libro" in user_question or "prestar" in user_question:
        return Answer(
            answer="Puedes tener hasta 5 libros prestados simultáneamente por 15 días",
            confidence=0.9
        )
    else:
        return Answer(
            answer="Solo puedo responder sobre el reglamento de la biblioteca. ¿Puedes reformular tu pregunta?",
            confidence=0.3,
            suggested_questions=[
                "¿Cuál es el horario de atención?",
                "¿Qué necesito para el carnet?",
                "¿Cómo reservo una sala?"
            ]
        )