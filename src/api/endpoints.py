from fastapi import APIRouter, HTTPException
from src.chatbot.core import ChatBot
from src.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
chatbot = ChatBot()

@router.post("/query", response_model=ChatResponse)
async def process_query(request: ChatRequest):
    """
    Endpoint principal para procesar preguntas del usuario
    """
    try:
        response = chatbot.process_question(request.question)
        return ChatResponse(
            question=request.question,
            answer=response["answer"],
            confidence=response["confidence"],
            source=response["source"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "library-chatbot"}