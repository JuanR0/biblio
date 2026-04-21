<<<<<<< Updated upstream
from fastapi import APIRouter, HTTPException
=======
import os
from fastapi import APIRouter, HTTPException, Request, Response, Header
>>>>>>> Stashed changes
from src.chatbot.core import ChatBot
from src.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
chatbot = ChatBot()

<<<<<<< Updated upstream
=======
ADMIN_TOKEN = os.environ.get("CHATBOT_ADMIN_TOKEN", "PassTest123")
ENABLE_ADMIN_ENDPOINTS = os.environ.get("ENABLE_ADMIN", "true").lower() == "true"

>>>>>>> Stashed changes
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
            source=response["source"],
            entities=response["entities"]
        )
    except Exception as e:
<<<<<<< Updated upstream
        raise HTTPException(status_code=500, detail=str(e))
=======
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

if ENABLE_ADMIN_ENDPOINTS:
    @router.post("/admin/reload")
    async def reload_knowledge(admin_token: str = Header(None, alias="Admin-Token")):
        """
        Recarga la base de conocimiento desde sin reiniciar el servicio.
        
        Requiere el header `Admin-Token` con el token de administración.
        """
        # Verificar el token (si está configurado)
        if ADMIN_TOKEN and admin_token != ADMIN_TOKEN:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "UNAUTHORIZED",
                    "message": "Token de administración inválido o ausente"
                }
            )
        
        try:
            # Recargar recursos
            chatbot.load_resources()
            
            # Obtener estadísticas actualizadas
            categories_info = {}
            for category in chatbot.category_keywords:
                knowledge = chatbot.knowledge_base.get_knowledge(category)
                if knowledge:
                    categories_info[category] = len(knowledge)
            
            return {
                "status": "success",
                "message": "Base de conocimiento recargada correctamente",
                "rules_loaded": categories_info,
                "synonyms_loaded": len(chatbot.matcher.synonyms) if hasattr(chatbot.matcher, 'synonyms') else 0
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "RELOAD_FAILED",
                    "message": f"Error al recargar: {str(e)}"
                }
            )
>>>>>>> Stashed changes

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "library-chatbot"}