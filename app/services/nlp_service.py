import spacy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from .synonyms import SynonymManager  # Nuevo import

class NLPService:
    def __init__(self):
        # Cargar modelo spaCy
        try:
            self.nlp = spacy.load("es_core_news_sm")
            print("✅ Modelo spaCy cargado")
        except:
            raise Exception("Modelo spaCy no encontrado. Ejecuta: python -m spacy download es_core_news_sm")
        
        # Inicializar gestor de sinónimos
        self.synonym_manager = SynonymManager()
        print("✅ Gestor de sinónimos inicializado")
        
        # Cargar conocimiento
        self.knowledge = self._load_knowledge()
        
    def _load_knowledge(self) -> Dict:
        """Carga los archivos JSON de conocimiento"""
        knowledge = {}
        data_dir = Path("data/knowledge")
        
        if data_dir.exists():
            for file_path in data_dir.glob("*.json"):
                if file_path.name == "synonyms.json":
                    continue  # Ya cargado por SynonymManager
                    
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
        question_lower = question.lower()
        doc = self.nlp(question_lower)
        
        # Extraer palabras clave (sin stopwords)
        keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        
        # Enriquecer con sinónimos
        enriched_keywords = self._enrich_with_synonyms(keywords, question_lower)
        
        # Buscar respuesta en conocimiento usando sinónimos
        answer, confidence, category = self._find_answer_with_synonyms(
            question_lower, 
            keywords, 
            enriched_keywords
        )
        
        # Extraer entidades nombradas
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        # Encontrar sinónimos detectados
        detected_synonyms = self.synonym_manager.find_keyword_synonyms(
            question_lower, 
            keywords
        )
        
        return {
            "answer": answer,
            "confidence": confidence,
            "entities": entities,
            "category": category,
            "keywords": keywords[:5],
            "enriched_keywords": list(enriched_keywords)[:10],
            "detected_synonyms": detected_synonyms
        }
    
    def _enrich_with_synonyms(self, keywords: List[str], question: str) -> Set[str]:
        """Enriquece las palabras clave con sus sinónimos"""
        enriched = set(keywords)
        
        for keyword in keywords:
            synonyms = self.synonym_manager.get_synonyms(keyword)
            enriched.update(synonyms)
        
        return enriched
    
    def _find_answer_with_synonyms(
        self, 
        question: str, 
        original_keywords: List[str], 
        enriched_keywords: Set[str]
    ) -> tuple:
        """Busca la mejor respuesta usando sinónimos"""
        best_answer = "Lo siento, solo puedo responder preguntas sobre el reglamento de la biblioteca."
        best_confidence = 0.1
        best_category = "general"
        
        for category, data in self.knowledge.items():
            for topic in data.get("topics", []):
                topic_keywords = topic.get("keywords", [])
                topic_confidence = topic.get("confidence", 0.8)
                
                # Calcular coincidencias considerando sinónimos
                matches = self._calculate_synonym_matches(
                    question, 
                    topic_keywords, 
                    enriched_keywords
                )
                
                if matches > 0:
                    # Base confidence + bonus por matches
                    confidence = min(
                        0.1 + (matches * 0.25) + (topic_confidence * 0.1),
                        0.95
                    )
                    
                    if confidence > best_confidence:
                        best_answer = topic["answer"]
                        best_confidence = confidence
                        best_category = category
        
        return best_answer, best_confidence, best_category
    
    def _calculate_synonym_matches(
        self, 
        question: str, 
        topic_keywords: List[str], 
        enriched_keywords: Set[str]
    ) -> int:
        """Calcula coincidencias considerando sinónimos"""
        matches = 0
        
        for topic_keyword in topic_keywords:
            # Verificar si la palabra o sus sinónimos están en la pregunta
            topic_keyword_synonyms = self.synonym_manager.get_synonyms(topic_keyword)
            
            # Verificar coincidencia directa
            if topic_keyword in question:
                matches += 2  # Mayor peso para coincidencia exacta
            
            # Verificar coincidencia con sinónimos enriquecidos
            elif any(syn in enriched_keywords for syn in topic_keyword_synonyms):
                matches += 1  # Peso menor para coincidencia por sinónimo
        
        return matches
    
    def get_synonym_suggestions(self, word: str) -> List[str]:
        """Obtiene sugerencias de sinónimos para una palabra"""
        synonyms = self.synonym_manager.get_synonyms(word)
        return list(synonyms)
    
    def analyze_synonym_match(self, question: str, target_word: str) -> Dict:
        """Analiza cómo se relaciona una palabra con sus sinónimos en la pregunta"""
        synonyms = self.synonym_manager.get_synonyms(target_word)
        question_lower = question.lower()
        
        found_words = []
        for synonym in synonyms:
            if synonym in question_lower:
                found_words.append(synonym)
        
        return {
            "target_word": target_word,
            "synonyms": list(synonyms),
            "found_in_question": found_words,
            "exact_match": target_word in question_lower,
            "synonym_match": len(found_words) > 0
        }