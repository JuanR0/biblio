import spacy
import json
from pathlib import Path
from typing import Dict, List, Optional

class NLPService:
    def __init__(self):
        # Cargar modelo spaCy
        try:
            self.nlp = spacy.load("es_core_news_sm")
            print("✅ Modelo spaCy cargado")
        except:
            raise Exception("Modelo spaCy no encontrado. Ejecuta: python -m spacy download es_core_news_sm")
        
        # Cargar conocimiento
        self.knowledge = self._load_knowledge()
        
    def _load_knowledge(self) -> Dict:
        """Carga los archivos JSON de conocimiento"""
        knowledge = {}
        data_dir = Path("data/knowledge")
        
        if data_dir.exists():
            for file_path in data_dir.glob("*.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    category = file_path.stem  # prestamos, horarios
                    knowledge[category] = json.load(f)
        else:
            # Conocimiento mínimo por defecto
            knowledge = {
                "préstamos": {
                    "category": "préstamos",
                    "topics": [
                        {
                            "topic": "límite",
                            "answer": "Puedes tener hasta 5 libros prestados simultáneamente.",
                            "keywords": ["libros", "máximo", "límite"]
                        }
                    ]
                }
            }
            
        return knowledge
    
    def process_question(self, question: str) -> Dict:
        """Procesa una pregunta y devuelve respuesta"""
        # Procesar con spaCy
        doc = self.nlp(question.lower())
        
        # Extraer palabras clave
        keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
        
        # Buscar respuesta en conocimiento
        answer, confidence, category = self._find_answer(question, keywords)
        
        # Extraer entidades nombradas
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        return {
            "answer": answer,
            "confidence": confidence,
            "entities": entities,
            "category": category,
            "keywords": keywords[:5]  # Top 5 keywords
        }
    
    def _find_answer(self, question: str, keywords: List[str]) -> tuple:
        """Busca la mejor respuesta en la base de conocimiento"""
        best_answer = "Lo siento, solo puedo responder preguntas sobre el reglamento de la biblioteca."
        best_confidence = 0.1
        best_category = "general"
        
        question_lower = question.lower()
        
        for category, data in self.knowledge.items():
            for topic in data.get("topics", []):
                # Buscar por palabras clave del tema
                topic_keywords = topic.get("keywords", [])
                
                # Calcular coincidencias
                matches = sum(1 for kw in topic_keywords if kw in question_lower)
                
                if matches > 0:
                    confidence = min(0.1 + (matches * 0.3), 0.95)
                    if confidence > best_confidence:
                        best_answer = topic["answer"]
                        best_confidence = confidence
                        best_category = category
        
        return best_answer, best_confidence, best_category