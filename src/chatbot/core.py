import json
import os
from typing import Dict, List, Tuple, Optional

# Intentar importar Spacy, pero tener fallback
try:
    import spacy
    SPACY_AVAILABLE = True
    print("✅ Spacy disponible")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  Spacy no disponible, usando sistema básico")

from .knowledge_base import KnowledgeBase
from .matcher import QueryMatcher

class ChatBot:
    def __init__(self, knowledge_path: str = "knowledge/", synonyms_path: str = "synonyms/", use_spacy: bool = True):
       
        self.knowledge_base = KnowledgeBase(knowledge_path)
        self.matcher = QueryMatcher(synonyms_path)
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # Inicializar Spacy si está disponible
        if self.use_spacy:
            self._init_spacy()
        
        self.load_resources()
        
        # Configuracion de categorias
        self.category_keywords = {
            "books": {
                "palabras": ["libro", "texto", "volumen", "obra", "lectura", "novela", 
                           "autor", "título", "editorial", "préstamo", "devolución",
                           "bibliografía", "referencia", "colección", "página"],
                "peso": 1.0,
                "exclusivas": ["libro", "texto", "volumen", "novela", "autor"]
            },
            "computers": {
                "palabras": ["computadora", "ordenador", "pc", "equipo", "software", 
                           "hardware", "internet", "impresora", "digital", "teclado",
                           "monitor", "programa", "aplicación", "red", "wifi", "online"],
                "peso": 1.0,
                "exclusivas": ["computadora", "ordenador", "pc", "software", "hardware"]
            },
            "cubicles": {
                "palabras": ["cubiculo", "sala", "espacio", "cabina", "estudio", 
                           "silencioso", "grupo", "reservar", "área", "individual",
                           "privado", "silenciosa", "trabajo", "concentración"],
                "peso": 1.0,
                "exclusivas": ["cubiculo", "cabina", "silencioso", "privado"]
            },
            "general": {
                "palabras": ["horario", "hora", "abrir", "cerrar", "baño", "wc", 
                           "servicio", "ubicación", "carné", "membresía", "impresión",
                           "wifi", "información", "ayuda", "contacto", "dirección",
                           "teléfono", "email", "normas", "reglamento", "acceso"],
                "peso": 0.7,
                "exclusivas": ["baño", "wc", "horario", "abrir", "cerrar"]
            }
        }
        
        self.debug_mode = True
    
    def _init_spacy(self):
        """Inicializa Spacy si está disponible"""
        try:
            
            self.nlp = spacy.load("es_core_news_sm")
            print("✅ Spacy inicializado con modelo 'es_core_news_sm'")
            print(f"📊 Pipeline disponible: {self.nlp.pipe_names}")
            
            # Configuracion para prevenir los errores de un inicio
            self._setup_spacy_features()
            
        except Exception as e:
            print(f"❌ Error inicializando Spacy: {e}")
            print("🔧 Continuando sin Spacy")
            self.use_spacy = False
    
    def _setup_spacy_features(self):
        """Configura Spacy"""
        # NO deshabilitar componentes, trabajar con lo que el modelo tiene
        print(f"📊 Usando pipeline existente: {self.nlp.pipe_names}")
        
        # El modelo español no tiene 'tagger' separado
        # Usa 'morphologizer' para POS tagging
        # El lematizador ya está incluido en el pipeline
        
        # Placeholder por si ocupamos deshabilitar componentes
        components_to_disable = []
        
        # Solo deshabilitar si existen en el pipeline
        for component in ["ner", "parser", "senter"]:
            if component in self.nlp.pipe_names:
                components_to_disable.append(component)
        
        if components_to_disable:
            for component in components_to_disable:
                self.nlp.disable_pipe(component)
            print(f"🔧 Componentes deshabilitados: {components_to_disable}")
            print(f"📊 Pipeline activo: {self.nlp.pipe_names}")
    
    def extract_lemmas_spacy(self, text: str) -> List[str]:
        """Extrae lemas usando Spacy"""
        if not self.use_spacy or not hasattr(self, 'nlp'):
            return []
        
        try:
            doc = self.nlp(text)
            lemmas = []
            
            for token in doc:
                # Filtrar stop words, puntuación y espacios
                if not token.is_stop and not token.is_punct and not token.is_space:
                    lemma = token.lemma_.lower().strip()
                    if lemma and len(lemma) > 2:
                        lemmas.append(lemma)
            
            return lemmas
            
        except Exception as e:
            print(f"⚠️  Error en extract_lemmas_spacy: {e}")
            return []
    
    def categorize_question(self, question: str) -> Tuple[str, float]:
        """
        Categoriza la pregunta usando texto NORMALIZADO primero
        """
        # Normalizacion de la pregunta primero que nada 
        question_normalized = self.matcher.normalize_text(question)
        
        if self.debug_mode:
            print(f"🔧 Pregunta normalizada: '{question}' → '{question_normalized}'")
        
        # Buscar palabras exclusivas en texto normalizado
        for category, data in self.category_keywords.items():
            exclusivas = data.get("exclusivas", [])
            for palabra in exclusivas:
                # Normalizar tambien la palabra exclusiva para comparar
                palabra_normalizada = self.matcher.normalize_text(palabra)
                if palabra_normalizada in question_normalized:
                    if self.debug_mode:
                        print(f"🔍 Categoría forzada a '{category}' por palabra exclusiva: '{palabra}' → '{palabra_normalizada}'")
                    return category, 1.0
        
        # Usar Spacy con texto normalizado 
        if self.use_spacy:
            try:
                # Para mejores reaultados se lematiza pregunta original
                lemmas = self.extract_lemmas_spacy(question)
                if lemmas:
                    if self.debug_mode:
                        print(f"🔍 Lemas Spacy: {lemmas}")
                    
                    # Buscar lemas en palabras clave
                    category_scores = {category: 0.0 for category in self.category_keywords}
                    
                    for category, data in self.category_keywords.items():
                        keywords = data["palabras"]
                        weight = data["peso"]
                        
                        # Normalizar cada keyword para comparar con lemas
                        keywords_normalizadas = [self.matcher.normalize_text(k) for k in keywords]
                        
                        for lemma in lemmas:
                            # Normalizar el lemma tambien
                            lemma_normalizado = self.matcher.normalize_text(lemma)
                            if lemma_normalizado in keywords_normalizadas:
                                category_scores[category] += weight
                                if self.debug_mode:
                                    print(f"   ✅ Lemma '{lemma}' → '{lemma_normalizado}' coincide con {category}")
                    
                    # Determinar ganador basado en lemas
                    best_category = max(category_scores, key=category_scores.get)
                    best_score = category_scores[best_category]
                    
                    if best_score > 0:
                        normalized_score = min(best_score / 5.0, 1.0)

                        if self.debug_mode:
                            print(f"🏆 Spacy seleccionó: {best_category} (score: {best_score}, conf: {normalized_score:.2f})")
                        
                        return best_category, normalized_score
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Error en categorización Spacy: {e}")
                # Continuar con metodo tradicional
        
        # Metodo tradicional con texto normalizado
        category_scores = {category: 0.0 for category in self.category_keywords}
        
        for category, data in self.category_keywords.items():
            keywords = data["palabras"]
            weight = data["peso"]
            
            # Normalizar cada keyword para comparar con pregunta normalizada
            keywords_normalizadas = [self.matcher.normalize_text(k) for k in keywords]
            
            for keyword, keyword_norm in zip(keywords, keywords_normalizadas):
                if keyword_norm in question_normalized:
                    # Bonus si la palabra aparece al inicio en texto normalizado
                    if question_normalized.startswith(keyword_norm + " "):
                        category_scores[category] += weight * 1.5
                        if self.debug_mode:
                            print(f"🚀 Bonus inicio: '{keyword}' → '{keyword_norm}' en {category}")

                    elif f" {keyword_norm} " in question_normalized:
                        category_scores[category] += weight
                        if self.debug_mode:
                            print(f"✅ Coincidencia: '{keyword}' → '{keyword_norm}' en {category}")

                    elif question_normalized.endswith(" " + keyword_norm):
                        category_scores[category] += weight
                        if self.debug_mode:
                            print(f"📍 Coincidencia final: '{keyword}' → '{keyword_norm}' en {category}")

                    else:
                        # Palabra como substring si se comio un espacio
                        category_scores[category] += weight * 0.7
                        if self.debug_mode:
                            print(f"🔍 Substring: '{keyword}' → '{keyword_norm}' en {category}")
        
        if self.debug_mode:
            print(f"\n📊 Puntuaciones finales: {category_scores}")
        
        # Determinar ganador
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        # Si no hay puntuación significativa, usar general
        if best_score < 0.5:
            if self.debug_mode:
                print(f"⚠️  Score bajo ({best_score:.2f}), usando 'general' por defecto")
            return "general", 0.5
        
        # Normalizar score
        max_possible = sum([data["peso"] * 3 for data in self.category_keywords.values()])
        normalized_score = min(best_score / max_possible, 1.0)
        
        if self.debug_mode:
            print(f"🏆 Ganador: {best_category} (score: {best_score:.2f}, confianza: {normalized_score:.2f})")
        
        return best_category, normalized_score

    def expand_query_with_spacy(self, query: str) -> List[str]:
        """Expande consultas usando lematización de Spacy - VERSIÓN SEGURA"""
        # Usar metodo tradicional
        traditional_expanded = self.matcher.expand_with_synonyms(query.lower())
        expanded_queries = list(set(traditional_expanded))  
        # Eliminar duplicados
        
        # Agregar lemas de Spacy si esta disponible
        if self.use_spacy and hasattr(self, 'nlp'):
            try:
                lemmas = self.extract_lemmas_spacy(query)
                if lemmas:
                    lemma_query = " ".join(lemmas)
                    normalized_query = self.matcher.normalize_text(query)
                    
                    if lemma_query != normalized_query and lemma_query not in expanded_queries:
                        expanded_queries.append(lemma_query)
                        
            except Exception as e:
                print(f"⚠️  Error en expand_query_with_spacy: {e}")
        
        return expanded_queries
    
    def calculate_similarity_enhanced(self, queries: List[str], target_phrases: List[str]) -> float:
        """Calcula similitud"""
        # Usar metodo tradicional
        traditional_similarity = self.matcher.calculate_similarity(queries, target_phrases)
        
        # Si Spacy no está disponible, retornar tradicional
        if not self.use_spacy or not hasattr(self, 'nlp'):
            return traditional_similarity
        
        try:
            # Intentar con Spacy
            max_similarity = 0.0
            
            for query in queries:
                if not query:
                    continue
                
                query_lemmas = set(self.extract_lemmas_spacy(query))
                
                for phrase in target_phrases:
                    if not phrase:
                        continue
                    
                    phrase_lemmas = set(self.extract_lemmas_spacy(phrase))
                    
                    # Similaridad Jaccard con lemas
                    if query_lemmas and phrase_lemmas:
                        intersection = len(query_lemmas & phrase_lemmas)
                        union = len(query_lemmas | phrase_lemmas)
                        
                        if union > 0:
                            lemma_similarity = intersection / union
                        else:
                            lemma_similarity = 0
                    else:
                        lemma_similarity = 0
                    
                    # Combinar con tradicional
                    combined = (lemma_similarity * 0.6) + (traditional_similarity * 0.4)
                    
                    if combined > max_similarity:
                        max_similarity = combined
            
            return max_similarity
            
        except Exception as e:
            print(f"⚠️  Error en calculate_similarity_enhanced: {e}")
            # Fallback a tradicional
            return traditional_similarity
    
    def load_resources(self):
        """Carga recursos"""
        try:
            self.knowledge_base.load_all_knowledge()
            self.matcher.load_synonyms()
            
            mode = "Spacy" if self.use_spacy else "Básico"
            print(f"✅ Chatbot inicializado en modo {mode}")
            
            # Mostrar estadísticas
            print("\n📊 ESTADISTICAS:")
            for category in ["general", "books", "computers", "cubicles"]:
                knowledge = self.knowledge_base.get_knowledge(category)
                if knowledge:
                    print(f"   {category}: {len(knowledge)} reglas")
                else:
                    print(f"   {category}: 0 reglas")
            
        except Exception as e:
            print(f"❌ Error cargando recursos: {e}")

            for category in ["general", "books", "computers", "cubicles"]:
                if not self.knowledge_base.get_knowledge(category):
                    self.knowledge_base.knowledge[category] = {}
    
    def search_in_category(self, category: str, expanded_queries: List[str]) -> Tuple[Optional[str], float]:
        """Busca en categoria"""
        best_answer = None
        best_confidence = 0.0
        
        knowledge = self.knowledge_base.get_knowledge(category)
        if not knowledge:
            return None, 0.0
        
        for key, data in knowledge.items():
            # Usar similitud mejorada si Spacy está disponible
            if self.use_spacy:
                confidence = self.calculate_similarity_enhanced(expanded_queries, data["preguntas"])
            else:
                confidence = self.matcher.calculate_similarity(expanded_queries, data["preguntas"])
            
            # # Boost para categorías específicas
            # if category != "general":
            #     confidence *= 1.2
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_answer = data["respuesta"]
        
        return best_answer, best_confidence
    
    def process_question(self, question: str) -> Dict:
        """Procesa pregunta usando texto normalizado"""
        # Validación básica
        if not question or not question.strip():
            return {
                "answer": "Haz una pregunta sobre los servicios de la biblioteca.",
                "confidence": 0.0,
                "source": "general",
                "mode": "basic" if not self.use_spacy else "spacy"
            }
        
        # Normalizar la pregunta al inicio para TODO el proceso
        question_for_processing = self.matcher.normalize_text(question)
        
        if self.debug_mode:
            print(f"\n{'='*60}")
            print(f"🤖 PROCESANDO: '{question}'")
            print(f"📝 Normalizado: '{question_for_processing}'")
            print(f"{'='*60}")
        
        # Categorizar con texto normalizado
        category, category_confidence = self.categorize_question(question_for_processing)
        
        if self.debug_mode:
            print(f"📋 Categoria: {category} (confianza: {category_confidence:.2f})")
        
        # Expandir consulta usando texto normalizado
        if self.use_spacy:
            expanded_queries = self.expand_query_with_spacy(question_for_processing)
        else:
            # Usar la pregunta normalizada
            expanded_queries = self.matcher.expand_with_synonyms(question_for_processing)
        
        if self.debug_mode and expanded_queries:
            print(f"🔍 Consultas expandidas ({len(expanded_queries)}): {expanded_queries[:3]}...")
        
        # Buscar en categoria principal
        best_answer, best_confidence = self.search_in_category(category, expanded_queries)
        best_source = category
        
        if self.debug_mode:
            print(f"🎯 Mejor coincidencia en {category}: {best_confidence:.3f}")
        
        # Umbrales por categoria
        category_thresholds = {
            "books": 0.4,
            "computers": 0.4,
            "cubicles": 0.4,
            "general": 0.3
        }
        
        threshold = category_thresholds.get(category, 0.4)
        
        # Si no supera umbral, buscar en general
        if best_confidence < threshold:
            if self.debug_mode:
                print(f"⚠️  Confianza baja ({best_confidence:.3f} < {threshold}), probando 'general'...")
            
            general_answer, general_confidence = self.search_in_category("general", expanded_queries)
            
            if general_confidence > best_confidence + 0.1:
                best_answer = general_answer
                best_confidence = general_confidence
                best_source = "general"
                
                if self.debug_mode:
                    print(f"🔄 Cambiando a 'general': {general_confidence:.3f}")
        
        # Usar respuesta de fallback 
        if best_confidence < 0.25:
            best_answer = self.get_fallback_response(best_source, question)
            best_confidence = 0.25
            
            if self.debug_mode:
                print(f"🆘 Usando respuesta de fallback")
        
        # Confianza final
        final_confidence = min(best_confidence * (0.5 + category_confidence * 0.5), 1.0)
        
        if self.debug_mode:
            print(f"\n📊 RESULTADO FINAL:")
            print(f"   Respuesta: {best_answer[:80]}...")
            print(f"   Confianza final: {final_confidence:.3f}")
            print(f"{'='*60}\n")
        
        return {
            "answer": best_answer,
            "confidence": round(final_confidence, 3),
            "source": best_source,
            "mode": "basic" if not self.use_spacy else "spacy",
            "details": {
                "category_confidence": round(category_confidence, 3),
                "expanded_queries_count": len(expanded_queries)
            }
        }
    
    def get_fallback_response(self, category: str, question: str = "") -> str:
        """Respuesta de fallback especifica por categoría"""
        
        question_lower = question.lower()
        
        if "libro" in question_lower or "texto" in question_lower:
            return "Para préstamos de libros, presenta tu carné en recepción. Se permiten hasta 3 libros por 15 días."

        if "cubiculo" in question_lower or "sala" in question_lower or "cabina" in question_lower:
            return "Para reservar cubículos, acércate a la recepción con tu carné. Se permite 1 reserva por día de máximo 3 horas."
        
        if "computadora" in question_lower or "ordenador" in question_lower:
            return "Las computadoras están disponibles por orden de llegada. Máximo 2 horas de uso. Presenta tu carné."
        
        # Respuestas genericas por categoria
        fallback_responses = {
            "books": "Información sobre libros disponible en recepción. Horario de atención: 8 AM a 6 PM.",
            "computers": "Consulta las reglas de uso de computadoras en el área de tecnología.",
            "cubicles": "Para reservar cubículos, visita la recepción con identificación.",
            "general": "No he entendido la pregunta, ¿podrías reformularla?."
        }
        
        return fallback_responses.get(category, fallback_responses["general"])
    
    def get_system_info(self) -> Dict:
        """Obtiene información del sistema"""
        categories_info = {}
        for category in self.category_keywords:
            knowledge = self.knowledge_base.get_knowledge(category)
            if knowledge:
                categories_info[category] = len(knowledge)
        
        spacy_info = {}
        if self.use_spacy and hasattr(self, 'nlp'):
            spacy_info = {
                "model": "es_core_news_sm",
                "pipeline": self.nlp.pipe_names,
                "vocab_size": len(self.nlp.vocab)
            }
        
        return {
            "mode": "spacy" if self.use_spacy else "basic",
            "spacy_available": SPACY_AVAILABLE,
            "spacy_enabled": self.use_spacy,
            "spacy_info": spacy_info,
            "rules_loaded": categories_info,
            "synonyms_loaded": len(self.matcher.synonyms) if hasattr(self.matcher, 'synonyms') else 0
        }


def create_chatbot(force_basic: bool = False) -> ChatBot:
    """Crea una instancia del chatbot"""
    
    use_spacy = SPACY_AVAILABLE and not force_basic
    
    if use_spacy:
        print("🚀 Creando chatbot con Spacy...")
    else:
        print("⚡ Creando chatbot en modo básico...")
        if SPACY_AVAILABLE and force_basic:
            print("📝 Nota: Spacy está disponible pero se forzó modo básico")
    
    return ChatBot(use_spacy=use_spacy)

# Funcin de diagnostico
def diagnose_spacy():
    """Diagnóstico de Spacy"""
    print("\n" + "="*60)
    print("🩺 DIAGNÓSTICO DE SPACY")
    print("="*60)
    
    if not SPACY_AVAILABLE:
        print("❌ Spacy no está instalado")
        print("💡 Solución: pip install spacy")
        return False
    
    print("✅ Spacy está instalado")
    
    try:
        nlp = spacy.load("es_core_news_sm")
        print("✅ Modelo 'es_core_news_sm' cargado correctamente")
        print(f"📊 Pipeline: {nlp.pipe_names}")
        print(f"📊 Vocabulario: {len(nlp.vocab)} palabras")
        
        # Probar procesamiento
        test_text = "reservar un cubículo"
        doc = nlp(test_text)
        print(f"\n🧪 Prueba con '{test_text}':")
        for token in doc:
            print(f"   '{token.text}' → Lemma: '{token.lemma_}', POS: '{token.pos_}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        print("💡 Solución: python -m spacy download es_core_news_sm")
        return False

if __name__ == "__main__":
    # Ejecutar diagnóstico
    diagnose_spacy()
    
    # Crear y probar chatbot
    print(f"\n{'='*60}")
    print("🤖 PRUEBA DEL CHATBOT")
    print("="*60)
    
    chatbot = create_chatbot()
    
    test_questions = [
        "¿cómo reservo un cubículo?",
        "multa por libro atrasado",
        "horario de la biblioteca",
        "¿dónde están los baños?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] Q: {question}")
        response = chatbot.process_question(question)
        print(f"    A: {response['answer'][:80]}...")
        print(f"    📍 {response['source']} | 🤖 {response['mode']} | 📊 {response['confidence']}")