import os
from typing import Optional

class Config:
    # ===== SERVIDOR =====
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # ===== APLICACIÓN =====
    APP_NAME: str = "Biblio Assistant"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Asistente virtual para biblioteca"
    
    # ===== NLP =====
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "es_core_news_sm")
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.3"))
    LANGUAGE: str = "es"
    
    # ===== EXTERNAL SERVICES =====
    MAIN_BACKEND_URL: str = os.getenv("MAIN_BACKEND_URL", "http://localhost:8000")
    
    # ===== LOGGING =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ===== VALIDACIONES =====
    def validate(self):
        """Valida configuraciones críticas"""
        if not 1 <= self.PORT <= 65535:
            raise ValueError(f"PORT inválido: {self.PORT}")
        if self.MIN_CONFIDENCE < 0 or self.MIN_CONFIDENCE > 1:
            raise ValueError(f"MIN_CONFIDENCE debe estar entre 0 y 1: {self.MIN_CONFIDENCE}")

# Crear instancia global
config = Config()

# Validar al importar
config.validate()