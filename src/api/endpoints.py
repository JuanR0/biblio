from fastapi import APIRouter, HTTPException, Request, Response
from src.chatbot.core import ChatBot
from src.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
chatbot = ChatBot()

@router.post("/query", response_model=ChatResponse)
async def process_query(request: ChatRequest, response: Response):
    """
    Endpoint principal para procesar preguntas del usuario
    Soporta sesiones via session_id o cookie
    """
    try:
        
        session_id = request.session_id
        
        # Procesa pregunta con sesion
        result = chatbot.process_question(request.question, session_id)
        
        # cookie para sesion
        if result.get('session_id'):
            response.set_cookie(
                key="session_id",
                value=result['session_id'],
                max_age=9000, 
                httponly=True,
                samesite="lax"
            )
        
        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            confidence=result["confidence"],
            source=result["source"],
            entities=result["entities"],
            session_id=result.get("session_id")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information (for debugging)"""
    session = chatbot.session_manager.get_session_summary(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    deleted = chatbot.session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.get("/stats")
async def get_stats():
    """Get session statistics"""
    return {
        "active_sessions": chatbot.session_manager.get_active_sessions_count(),
        "session_timeout": chatbot.session_manager.session_timeout
    }