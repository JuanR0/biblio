from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .api.endpoints import chat
from .core.config import config

app = FastAPI(debug=config.DEBUG)

# Crear la aplicación FastAPI
app = FastAPI(
    title=config.APP_NAME,
    debug=config.DEBUG,
    version=config.VERSION
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(chat.router, prefix="/api", tags=["chat"])

# Endpoint raíz
@app.get("/")
def root():
    return {"message": "Biblio Assistant API", "docs": "/docs"}

# Health check
@app.get("/health")
def health():
    return {"status": "healthy", "service": "biblio-assistant"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=config.PORT,  # ← Usas config.PORT
        reload=config.DEBUG  # ← Usas config.DEBUG
    )