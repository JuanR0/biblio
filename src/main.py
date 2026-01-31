from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import router as chatbot_router

app = FastAPI(title="Library Chatbot Microservice")

# Configurar CORS para permitir conexión desde el backend principal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # URL de tu backend principal
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router)

@app.get("/")
def read_root():
    return {"status": "Chatbot microservice running"}