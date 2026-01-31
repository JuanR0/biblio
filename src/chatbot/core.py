import json
import os
from typing import Dict, List, Tuple, Optional
from .knowledge_base import KnowledgeBase
from .matcher import QueryMatcher

class ChatBot:
    def __init__(self, knowledge_path: str = "knowledge/", synonyms_path: str = "synonyms/"):
        """
        Inicializa el chatbot con las rutas a los archivos de conocimiento
        """
        self.knowledge_base = KnowledgeBase(knowledge_path)
        self.matcher = QueryMatcher(synonyms_path)
        self.load_resources()
        
        # Palabras clave por categoría para clasificación
        self.category_keywords = {
            "books": {
                "palabras": ["libro", "texto", "volumen", "obra", "lectura", "novela", 
                           "autor", "título", "editorial", "préstamo", "devolución",
                           "bibliografía", "referencia", "colección", "página"],
                "peso": 1.0,
                "exclusivas": ["libro", "texto", "volumen", "novela", "autor"]  # Palabras que SIEMPRE indican esta categoría
            },
            "computers": {
                "palabras": ["computadora", "ordenador", "pc", "equipo", "software", 
                           "hardware", "internet", "impresora", "digital", "teclado",
                           "monitor", "programa", "aplicación", "red", "wifi", "online",
                           "tecnología", "dispositivo"],
                "peso": 1.0,
                "exclusivas": ["computadora", "ordenador", "pc", "software", "hardware"]
            },
            "cubicles": {
                "palabras": ["cubiculo", "sala", "espacio", "cabina", "estudio", 
                           "silencioso", "grupo", "reservar", "área", "individual",
                           "privado", "silenciosa", "trabajo", "concentración", 
                           "apartar", "lugar", "habitación"],
                "peso": 1.2,  # Mayor peso para cubículos (menos comunes)
                "exclusivas": ["cubiculo", "cabina", "silencioso", "privado"]
            },
            "general": {
                "palabras": ["horario", "hora", "abrir", "cerrar", "baño", "wc", 
                           "servicio", "ubicación", "carné", "membresía", "impresión",
                           "wifi", "información", "ayuda", "contacto", "dirección",
                           "teléfono", "email", "normas", "reglamento", "acceso",
                           "general", "precio", "costo", "tarifa"],
                "peso": 0.7,
                "exclusivas": ["baño", "wc", "horario", "abrir", "cerrar"]
            }
        }
        
        self.debug_mode = True  # Cambiar a False en producción
    
    def load_resources(self):
        """Carga todos los archivos de conocimiento y sinónimos"""
        try:
            self.knowledge_base.load_all_knowledge()
            self.matcher.load_synonyms()
            
            # Verificar que todos los archivos se cargaron
            for category in ["general", "books", "computers", "cubicles"]:
                knowledge = self.knowledge_base.get_knowledge(category)
                if knowledge:
                    print(f"✅ {category}: {len(knowledge)} reglas cargadas")
                else:
                    print(f"⚠️  {category}: No se cargaron reglas")
                    
        except Exception as e:
            print(f"❌ Error cargando recursos: {e}")
            raise
    
    def categorize_question(self, question: str) -> Tuple[str, float]:
        """
        Identifica la categoría más probable de la pregunta
        con reglas estrictas para palabras exclusivas
        """
        question_lower = question.lower()
        
        # PRIMERO: Buscar palabras exclusivas (categoría forzada)
        for category, data in self.category_keywords.items():
            exclusivas = data.get("exclusivas", [])
            for palabra in exclusivas:
                if palabra in question_lower:
                    if self.debug_mode:
                        print(f"🔍 Categoría forzada a '{category}' por palabra exclusiva: '{palabra}'")
                    return category, 1.0  # Máxima confianza
        
        # SEGUNDO: Sistema de puntuación normal
        category_scores = {category: 0.0 for category in self.category_keywords}
        
        for category, data in self.category_keywords.items():
            keywords = data["palabras"]
            weight = data["peso"]
            
            for keyword in keywords:
                if keyword in question_lower:
                    # Bonus si la palabra aparece al inicio
                    if question_lower.startswith(keyword + " ") or f" {keyword} " in question_lower:
                        category_scores[category] += weight * 1.5
                    else:
                        category_scores[category] += weight
        
        # TERCERO: Frases compuestas (boost alto)
        phrases_boost = {
            "books": [
                "prestar libro", "tomar prestado", "devolver libro", 
                "multa libro", "renovar libro", "préstamo libro"
            ],
            "computers": [
                "usar computadora", "reservar computadora", "acceder a pc",
                "tiempo computadora", "software biblioteca"
            ],
            "cubicles": [
                "reservar cubiculo", "sala estudio", "cabina individual",
                "espacio silencioso", "área estudio", "lugar concentración"
            ],
            "general": [
                "dónde está", "qué hora", "cuánto cuesta", "cómo obtener",
                "hora atención", "teléfono biblioteca"
            ]
        }
        
        for category, phrases in phrases_boost.items():
            for phrase in phrases:
                if phrase in question_lower:
                    category_scores[category] += 3.0  # Boost muy alto por frase exacta
                    if self.debug_mode:
                        print(f"🚀 Boost +3.0 a '{category}' por frase: '{phrase}'")
        
        # Determinar ganador
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        if self.debug_mode:
            print(f"📊 Puntuaciones: {category_scores}")
            print(f"🏆 Ganador: {best_category} ({best_score:.2f})")
        
        # Si no hay puntuación significativa, usar general
        if best_score < 0.5:
            return "general", 0.5
        
        # Normalizar a 0-1
        total_possible = sum([data["peso"] * 3 for data in self.category_keywords.values()])
        normalized_score = min(best_score / total_possible, 1.0)
        
        return best_category, normalized_score
    
    def search_in_category(self, category: str, expanded_queries: List[str], original_question: str = "") -> Tuple[Optional[str], float, Dict]:
        """
        Busca la mejor respuesta dentro de una categoría específica
        Devuelve también detalles de debugging
        """
        best_answer = None
        best_confidence = 0.0
        best_match_details = {}
        
        knowledge = self.knowledge_base.get_knowledge(category)
        if not knowledge:
            if self.debug_mode:
                print(f"❌ No hay conocimiento para categoría: {category}")
            return None, 0.0, {}
        
        if self.debug_mode:
            print(f"🔎 Buscando en categoría: {category}")
            print(f"   Consultas expandidas: {expanded_queries[:3]}...")
        
        for key, data in knowledge.items():
            confidence = self.matcher.calculate_similarity(
                expanded_queries, 
                data["preguntas"]
            )
            
            # Aumentar confianza si es la categoría correcta
            if category != "general":
                confidence *= 1.3  # Boost del 30% para categorías específicas
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_answer = data["respuesta"]
                best_match_details = {
                    "rule_key": key,
                    "matched_questions": data["preguntas"],
                    "raw_confidence": confidence
                }
                
                if self.debug_mode and confidence > 0.3:
                    print(f"   ✅ Regla '{key}': confianza={confidence:.3f}")
        
        if self.debug_mode:
            print(f"   🎯 Mejor confianza en {category}: {best_confidence:.3f}")
        
        return best_answer, best_confidence, best_match_details
    
    def process_question(self, question: str) -> Dict:
        """
        Procesa una pregunta del usuario y devuelve la respuesta más apropiada
        """
        if self.debug_mode:
            print(f"\n{'='*60}")
            print(f"🤖 PROCESANDO PREGUNTA: '{question}'")
            print(f"{'='*60}")
        
        # Validar entrada
        if not question or not question.strip():
            return {
                "answer": "Por favor, formula una pregunta sobre los servicios de la biblioteca.",
                "confidence": 0.0,
                "source": "general"
            }
        
        # 1. Categorizar la pregunta
        category, category_confidence = self.categorize_question(question)
        
        if self.debug_mode:
            print(f"📋 Categoría identificada: {category} (confianza: {category_confidence:.2f})")
        
        # 2. Expandir la pregunta con sinónimos
        expanded_queries = self.matcher.expand_with_synonyms(question.lower())
        
        # 3. Buscar SOLO en la categoría identificada (estrategia estricta)
        best_answer, best_confidence, match_details = self.search_in_category(
            category, expanded_queries, question
        )
        best_source = category
        
        # 4. NUEVA ESTRATEGIA: Umbrales diferentes por categoría
        category_thresholds = {
            "books": 0.4,
            "computers": 0.4,
            "cubicles": 0.35,  # Umbral más bajo para cubículos (menos preguntas)
            "general": 0.3
        }
        
        threshold = category_thresholds.get(category, 0.4)
        
        # 5. Si no supera el umbral, buscar en general como fallback
        if best_confidence < threshold:
            if self.debug_mode:
                print(f"⚠️  Confianza baja ({best_confidence:.3f} < {threshold}), probando 'general'...")
            
            general_answer, general_confidence, general_details = self.search_in_category(
                "general", expanded_queries, question
            )
            
            # Usar general solo si es significativamente mejor
            if general_confidence > best_confidence + 0.1:  # 10% mejor
                best_answer = general_answer
                best_confidence = general_confidence
                best_source = "general"
                match_details = general_details
                
                if self.debug_mode:
                    print(f"   🔄 Cambiando a 'general': {general_confidence:.3f}")
        
        # 6. Si aún no hay buena respuesta, usar fallback específico
        if best_confidence < 0.25:  # Umbral muy bajo
            best_answer = self.get_fallback_response(best_source, question)
            best_confidence = 0.25
            if self.debug_mode:
                print(f"   🆘 Usando respuesta de fallback para {best_source}")
        
        # 7. Ajustar confianza final
        final_confidence = min(best_confidence * (0.5 + category_confidence * 0.5), 1.0)
        
        # 8. Debugging detallado
        if self.debug_mode:
            print(f"\n📊 RESULTADO FINAL:")
            print(f"   Categoría: {best_source}")
            print(f"   Confianza: {final_confidence:.3f}")
            print(f"   Respuesta: {best_answer[:80]}...")
            
            if match_details:
                print(f"\n🔧 DETALLES DE COINCIDENCIA:")
                print(f"   Regla: {match_details.get('rule_key', 'N/A')}")
                print(f"   Confianza cruda: {match_details.get('raw_confidence', 0):.3f}")
            
            print(f"\n{'='*60}\n")
        
        return {
            "answer": best_answer,
            "confidence": round(final_confidence, 3),
            "source": best_source,
            "details": {
                "category_confidence": round(category_confidence, 3),
                "threshold_used": threshold,
                "matched_rule": match_details.get("rule_key", "fallback"),
                "debug": self.debug_mode
            } if self.debug_mode else None
        }
    
    def get_fallback_response(self, category: str, question: str = "") -> str:
        """
        Proporciona una respuesta de fallback específica para cada categoría
        """
        # Respuestas CORREGIDAS - cada una específica para su categoría
        fallback_responses = {
            "books": "Para información detallada sobre préstamos, renovaciones y multas de libros, por favor consulta en la recepción de la biblioteca con tu carné de estudiante vigente.",
            "computers": "El uso de computadoras tiene reglas específicas de reserva y tiempo límite. Acércate al área de tecnología o consulta en recepción para conocer los detalles.",
            "cubicles": "La reserva de cubículos se realiza personalmente en la recepción. Cada estudiante puede reservar máximo 3 horas por día. Trae tu identificación.",
            "general": "¿Podrías especificar más tu pregunta? Si necesitas ayuda inmediata, el personal en recepción estará encantado de asistirte."
        }
        
        # Si detectamos "cubículo" en la pregunta, forzar respuesta de cubículos
        question_lower = question.lower()
        if any(word in question_lower for word in ["cubiculo", "cabina", "sala estudio"]):
            return fallback_responses["cubicles"]
        
        return fallback_responses.get(category, fallback_responses["general"])