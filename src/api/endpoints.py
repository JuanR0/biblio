from fastapi import APIRouter, HTTPException, Request, Response
from src.chatbot.core import ChatBot
from src.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
chatbot = ChatBot()


@router.post("/query", response_model=ChatResponse)
async def process_query(request: ChatRequest, response: Response, http_request: Request):
    """Endpoint principal para procesar preguntas del usuario"""
    
    # Validar session_id recibida por programa principal
    if not request.session_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "MISSIN_SESSION_ID",
                "message": "El programa principal debe proveer de session_id para crear la sesion"
            }
        )

    try:
        # Validacion de id para sesion, 
        result = chatbot.process_question(request.question, request.session_id)
        
        if result.get("error") == "missing_session_id":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "SESSION_ID_REQUIRED",
                    "message": result["answer"]
                }
            )
        
        # Manejo de limite de preguntas
        if result.get("rate_limited"):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "TOO_MANY_REQUESTS",
                    "wait_seconds": result.get("wait_time"),
                    "message": result["answer"]
                }
            )
        
        # Rate limit headers para monitoreo
        response.headers["X-RateLimit-Limit"] = str(chatbot.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(result.get("rate_limit_remaining", 2))
        response.headers["X-RateLimit-Window"] = str(chatbot.rate_limiter.window_seconds)
        
        # Session header para debugging
        response.headers["X-Session-ID"] = result["session_id"]
        
        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            confidence=result["confidence"],
            source=result["source"],
            entities=result["entities"],
            session_id=result["session_id"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "INTERNAL_ERROR",
                "message": str(e)
            }
        )


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Recuperar informacion de la sesion"""
    
    session = chatbot.session_manager.get_session_summary(session_id)
    if not session:
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Ninguna sesion existe con el ID: {session_id}"
            }
        )
    return session


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Eliminar sesion
    Para no depender exclusivamente de timeout
    """
    deleted = chatbot.session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Cannot delete: session {session_id} does not exist"
            }
        )
    return {
        "status": "deleted", 
        "session_id": session_id,
        "message": "Session successfully deleted"
    }


@router.delete("/sessions/all")
async def delete_all_sessions():
    """Borrado de todas las sesiones"""
    count = chatbot.session_manager.delete_all_sessions()
    return {
        "status": "deleted",
        "sessions_deleted": count,
        "message": f"All {count} sessions have been deleted"
    }


@router.get("/stats")
async def get_stats():
    """Obtener las estadisticas de sesion"""
    active_sessions = chatbot.session_manager.get_active_sessions_count()
    
    # Si se tiene estadisticas disponibles
    rate_limit_stats = {}
    if hasattr(chatbot, 'rate_limit_violations'):
        rate_limit_stats["total_violations"] = chatbot.rate_limit_violations
    if hasattr(chatbot, 'rate_limiter'):
        rate_limit_stats["max_requests"] = chatbot.rate_limiter.max_requests
        rate_limit_stats["window_seconds"] = chatbot.rate_limiter.window_seconds
    
    return {
        "active_sessions": active_sessions,
        "session_timeout_seconds": chatbot.session_manager.session_timeout,
        "rate_limiter": {
            "max_requests_per_window": chatbot.rate_limiter.max_requests,
            "window_seconds": chatbot.rate_limiter.window_seconds
        },
        "rate_limit_stats": rate_limit_stats,
        "chatbot_mode": "spacy" if chatbot.use_spacy else "basic"
    }


@router.get("/health")
async def health_check():
    """Endpoint para validar comunicacion"""
    return {
        "status": "healthy",
        "service": "library_chatbot",
        "session_count": chatbot.session_manager.get_active_sessions_count(),
        "mode": "spacy" if chatbot.use_spacy else "basic"
    }