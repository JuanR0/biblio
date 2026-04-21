from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import router as chatbot_router

app = FastAPI(title="Library Chatbot Microservice")

# Placeholder CORS, para permitir conexión desde el programa principal principal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router)

@app.get("/")
def read_root():
    return {"status": "Chatbot microservice running"}