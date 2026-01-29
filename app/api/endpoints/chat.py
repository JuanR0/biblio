from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.services.nlp_service import NLPService

router = APIRouter()

# Dependencia para el servicio NLP
def get_nlp_service():
    return NLPService()

# Modelos
class Question(BaseModel):
    question: str
    user_id: Optional[str] = None

class Answer(BaseModel):
    answer: str
    confidence: float
    entities: List[dict]
    category: Optional[str] = None
    suggested_questions: List[str] = []

# Endpoint principal
@router.post("/ask", response_model=Answer)
async def ask_question(
    question: Question,
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """Procesa preguntas usando NLP"""
    
    # Procesar con NLP
    result = nlp_service.process_question(question.question)
    
    # Generar preguntas sugeridas basadas en la categoría
    suggested = []
    if result["confidence"] > 0.5:
        if result["category"] == "préstamos":
            suggested = [
                "¿Cuál es la duración del préstamo?",
                "¿Cómo puedo renovar un libro?",
                "¿Hay multas por retraso?"
            ]
        elif result["category"] == "horarios":
            suggested = [
                "¿Abren los sábados?",
                "¿Cuál es el horario de cierre?",
                "¿Atienden en feriados?"
            ]
    
    return Answer(
        answer=result["answer"],
        confidence=result["confidence"],
        entities=result["entities"],
        category=result["category"],
        suggested_questions=suggested
    )

# Endpoint de prueba del NLP
@router.get("/test-nlp/{text}")
async def test_nlp(
    text: str,
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """Endpoint para probar el procesamiento NLP"""
    result = nlp_service.process_question(text)
    return {
        "input": text,
        "processed": result
    }

@router.get("/synonyms/{word}")
async def get_synonyms(
    word: str,
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """Obtiene sinónimos de una palabra"""
    synonyms = nlp_service.get_synonym_suggestions(word)
    return {
        "word": word,
        "synonyms": synonyms,
        "count": len(synonyms)
    }

@router.post("/analyze-synonyms")
async def analyze_synonyms(
    question: str,
    word: str,
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """Analiza cómo se relaciona una palabra con sus sinónimos en una pregunta"""
    analysis = nlp_service.analyze_synonym_match(question, word)
    return analysis