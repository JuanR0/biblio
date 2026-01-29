from fastapi import FastAPI
from app.api.endpoints import chat
from core.config import config

app = FastAPI(
    title=config.APP_NAME,
    debug=config.DEBUG,
    version="1.0.0"
)

# Incluir routers
app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/")
def root():
    return {
        "app": config.APP_NAME,
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/ask",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": config.APP_NAME,
        "nlp_ready": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=config.DEBUG
    )