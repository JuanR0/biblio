import json
import os
from typing import Dict, List, Tuple, Optional
from .session_manager import SessionManager
from .rate_limiter import RateLimiter, TieredRateLimiter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle  # optional, if you want to save/load model later

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

        fallback_threshold: float = 0.25
        log_low_confidence: bool = False
        low_confidence_log_path: str = "logs/low_confidence_queries.log"

        enable_ml_classifier: bool = True
        ml_confidence_threshold: float = 0.35   # Rule confidence below this triggers ML
        ml_override_threshold: float = 0.65 

        self.enable_ml_classifier = enable_ml_classifier
        self.ml_confidence_threshold = ml_confidence_threshold
        self.ml_override_threshold = ml_override_threshold

        self.vectorizer = None
        self.classifier = None
        self.ml_ready = False

        self.debug_mode = False

        #Initialize session manager
        self.session_manager = SessionManager(session_timeout=1800)

        self.rate_limiter = RateLimiter(
                    max_requests=2,      # 2 solicitudes
                    window_seconds=10     # cada 10 segundos
                )

        self.rate_limit_violations = 0

        # Inicializar Spacy si está disponible
        if self.use_spacy:
            self._init_spacy()
        
        self.load_resources()
        
        self.fallback_threshold = fallback_threshold
        self.log_low_confidence = log_low_confidence
        self.low_confidence_log_path = low_confidence_log_path
        
        if self.enable_ml_classifier:
            self._train_classifier()
        
        # Crear directorio de logs si está habilitado
        if self.log_low_confidence:
            os.makedirs(os.path.dirname(low_confidence_log_path), exist_ok=True)

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
            },
            "biblio": {
                "palabras": [
                    "biblio", "asistente", "ayuda", "información", "qué haces", 
                    "quién eres", "presentación", "bienvenida", "funciones", 
                    "servicios", "comandos", "preguntar", "como usar", "instrucciones",
                    "acerca de", "sobre ti", "capacidades", "qué puedes hacer"
                ],
                "peso": 0.7,
                "exclusivas": ["biblio", "asistente", "qué haces", "quién eres"]
            }
        }
    
    def _init_spacy(self):
        """Inicializa Spacy si está disponible"""
        try:
            
            self.nlp = spacy.load("es_core_news_sm")
            print("✅ Spacy inicializado con modelo 'es_core_news_sm'")
            print(f" Pipeline disponible: {self.nlp.pipe_names}")
            
            # Configuracion para prevenir los errores de un inicio
            self._setup_spacy_features()
            
        except Exception as e:
            print(f"❌ Error inicializando Spacy: {e}")
            print(" Continuando sin Spacy")
            self.use_spacy = False
    
    def _setup_spacy_features(self):
        """Configura Spacy"""
        # NO deshabilitar componentes, trabajar con lo que el modelo tiene
        print(f" Usando pipeline existente: {self.nlp.pipe_names}")
        
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
            print(f" Componentes deshabilitados: {components_to_disable}")
            print(f" Pipeline activo: {self.nlp.pipe_names}")
    
    def check_rate_limit(self, identifier: str) -> tuple:
        """
        Asegurarse de que el ID puede hacer preguntas.
        
        Args:
            identifier: User ID/session ID
            
        Returns:
            (allowed, wait_time, remaining)
            - allowed: Se permite
            - wait_time: Segundos de enfriamiento si no
            - remaining: solicitudes restantes
        """
        allowed, result = self.rate_limiter.is_allowed(identifier)
        
        if allowed:
            return True, None, result  # result = solicitudes restantes
        else:
            self.rate_limit_violations += 1
            return False, result, 0    # result = segundos de espera

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
    
    def categorize_question(self, question: str, context_category: str = None) -> Tuple[str, float]:
        """
        Categoriza la pregunta usando texto NORMALIZADO y contexto de sesión
        Args:
            question: La pregunta del usuario (normalizada)
            context_category: Categoría anterior de la sesión (opcional)
        Returns:
            Tuple[str, float]: (categoría, confianza)
        """
        # Normalizacion de la pregunta 
        question_normalized = self.matcher.normalize_text(question)
        
        if self.debug_mode:
            print(f" Pregunta normalizada: '{question}' → '{question_normalized}'")
            if context_category:
                print(f" Contexto: categoría anterior = '{context_category}'")
        
        # Buscar palabras exclusivas (ALTA CONFIANZA)
        for category, data in self.category_keywords.items():
            exclusivas = data.get("exclusivas", [])
            for palabra in exclusivas:
                palabra_normalizada = self.matcher.normalize_text(palabra)
                if palabra_normalizada in question_normalized:
                    if self.debug_mode:
                        print(f" Categoría forzada a '{category}' por palabra exclusiva: '{palabra}' → '{palabra_normalizada}'")
                    return category, 1.0
        
        # Spacy para lematización (si está disponible) 
        spacy_score = 0.0
        spacy_category = None
        
        if self.use_spacy:
            try:
                lemmas = self.extract_lemmas_spacy(question)
                if lemmas:
                    if self.debug_mode:
                        print(f" Lemas Spacy: {lemmas}")
                    
                    category_scores = {category: 0.0 for category in self.category_keywords}
                    
                    for category, data in self.category_keywords.items():
                        keywords = data["palabras"]
                        weight = data["peso"]
                        
                        keywords_normalizadas = [self.matcher.normalize_text(k) for k in keywords]
                        
                        for lemma in lemmas:
                            lemma_normalizado = self.matcher.normalize_text(lemma)
                            if lemma_normalizado in keywords_normalizadas:
                                category_scores[category] += weight
                                if self.debug_mode:
                                    print(f"   ✅ Lemma '{lemma}' → '{lemma_normalizado}' coincide con {category}")
                    
                    best_category = max(category_scores, key=category_scores.get)
                    best_score = category_scores[best_category]
                    
                    if best_score > 0:
                        spacy_category = best_category
                        spacy_score = min(best_score / 5.0, 1.0)
                        
                        if self.debug_mode:
                            print(f" Spacy seleccionó: {best_category} (score: {best_score}, conf: {spacy_score:.2f})")
                            
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Error en categorización Spacy: {e}")
        
        # Método tradicional con palabras clave
        category_scores = {category: 0.0 for category in self.category_keywords}
        
        for category, data in self.category_keywords.items():
            keywords = data["palabras"]
            weight = data["peso"]
            
            keywords_normalizadas = [self.matcher.normalize_text(k) for k in keywords]
            
            for keyword, keyword_norm in zip(keywords, keywords_normalizadas):
                if keyword_norm in question_normalized:
                    # Bonus si la palabra aparece al inicio
                    if question_normalized.startswith(keyword_norm + " "):
                        category_scores[category] += weight * 1.5
                        if self.debug_mode:
                            print(f" Bonus inicio: '{keyword}' → '{keyword_norm}' en {category}")
                    
                    elif f" {keyword_norm} " in question_normalized:
                        category_scores[category] += weight
                        if self.debug_mode:
                            print(f"✅ Coincidencia: '{keyword}' → '{keyword_norm}' en {category}")
                    
                    elif question_normalized.endswith(" " + keyword_norm):
                        category_scores[category] += weight
                        if self.debug_mode:
                            print(f" Coincidencia final: '{keyword}' → '{keyword_norm}' en {category}")
                    
                    else:
                        # Palabra como substring
                        category_scores[category] += weight * 0.7
                        if self.debug_mode:
                            print(f"🔍 Substring: '{keyword}' → '{keyword_norm}' en {category}")
        
        if self.debug_mode:
            print(f"\n📊 Puntuaciones tradicionales: {category_scores}")
        
        # Combinar resultados de Spacy y método tradicional
        best_category = None
        best_score = 0.0
        
        if spacy_category and spacy_score > 0:
            # Combinar Spacy (60%) con tradicional (40%)
            traditional_score = category_scores.get(spacy_category, 0)
            combined_score = (spacy_score * 0.6) + (min(traditional_score / 10.0, 1.0) * 0.4)
            
            best_category = spacy_category
            best_score = combined_score
            
            if self.debug_mode:
                print(f" Combinado Spacy: {spacy_category} (score: {combined_score:.2f})")
        else:
            # Solo método tradicional
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
        
        # Normalizar score 
        max_possible = sum([data["peso"] * 3 for data in self.category_keywords.values()])
        normalized_score = min(best_score / max_possible, 1.0)
        
        if self.debug_mode:
            print(f" Score normalizado: {normalized_score:.2f}")
        
        # Verificar si necesita usar contexto
        # Si la confianza es baja Y tenemos contexto de sesión
        if context_category and normalized_score < 0.5:
            
            # Palabras que indican seguimiento de conversación
            followup_indicators = [
                "y", "tambien", "ademas", "entonces", "eso", "esa", "ese",
                "y como", "y cuando", "y donde", "y que", "y cuanto",
                "entonces como", "entonces cuando", "entonces donde",
                "cual", "que", "cuanto", "como", "donde", "cuando"
            ]
            
            # Detectar si es pregunta de seguimiento
            is_followup = False
            question_lower = question_normalized.lower()

            is_short = len(question_normalized.split()) <= 6

            for indicator in followup_indicators:
                if question_lower.startswith(indicator) or f" {indicator} " in question_lower:
                    is_followup = True
                    break
            
            # preguntas muy cortas (probablemente seguimiento)
            if len(question_normalized.split()) <= 5:
                is_followup = True
            
            if is_followup or is_short:
                if self.debug_mode:
                    print(f"📌 Detectada pregunta de seguimiento (score bajo: {normalized_score:.2f})")
                    print(f"   Usando contexto de sesión: '{context_category}'")
                
                # Usar la categoría del contexto con confianza moderada
                return context_category, 0.65
        
        # Si no hay puntuación significativa, usar general 
        if normalized_score < 0.3:
            if self.debug_mode:
                print(f"⚠️  Score bajo ({normalized_score:.2f}), usando 'general' por defecto")
            return "general", 0.5
        
        if self.debug_mode:
            print(f" Ganador: {best_category} (confianza: {normalized_score:.2f})")
        
        return best_category, normalized_score
        
    def extract_entities(self, question: str) -> Dict[str, List[str]]:
        """
        Extraccion de entidades especificas de biblioteca.
        """
        # Normalizacion y lematizacion de la prgunta
        question_norm = self.matcher.normalize_text(question)
        lemmas = []
        if self.use_spacy and hasattr(self, 'nlp'):
            try:
                doc = self.nlp(question)
                lemmas = [token.lemma_.lower() for token in doc 
                        if not token.is_stop and not token.is_punct]
                if self.debug_mode:
                    print(f"🔍 Lemmas: {lemmas}")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error de lemmatizacion: {e}")

        words = question_norm.split()
        
        entities = {
            "locations": [],
            "resources": [],
            "actions": [],
            "time_related": [],
            "urgency": []
        }
        
        # Formas base para el mapeo de entidad 
        location_keywords = {"cubiculo", "sala", "cabina", "espacio", "area"}
        resource_keywords = {"libro", "computadora", "ordenador", "texto", "obra", "volumen"}
        action_keywords = {"reservar", "prestar", "devolver", "usar", "utilizar", "solicitar"}
        time_keywords = {"hora", "dia", "plazo", "tiempo", "minuto"}
        urgency_keywords = {"urgente", "problema", "multa", "cobro", "error", "emergencia"}
        
        # Comparacion de palabras 
        all_words_to_check = words + lemmas 
        
        for word in all_words_to_check:
            word_lower = word.lower()
            
            # Checar forma base
            if word_lower in location_keywords and word_lower not in entities["locations"]:
                entities["locations"].append(word_lower)
            elif word_lower in resource_keywords and word_lower not in entities["resources"]:
                entities["resources"].append(word_lower)
            elif word_lower in action_keywords and word_lower not in entities["actions"]:
                entities["actions"].append(word_lower)
            elif word_lower in time_keywords and word_lower not in entities["time_related"]:
                entities["time_related"].append(word_lower)
            elif word_lower in urgency_keywords and word_lower not in entities["urgency"]:
                entities["urgency"].append(word_lower)
        
        return {k: v for k, v in entities.items() if v}

    def expand_query_with_spacy(self, query: str) -> List[str]:
        """Expande consultas usando lematización de Spacy"""
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
            print("\n ESTADISTICAS:")
            for category in ["general", "books", "computers", "cubicles", "biblio"]:
                knowledge = self.knowledge_base.get_knowledge(category)
                if knowledge:
                    print(f"   {category}: {len(knowledge)} reglas")
                else:
                    print(f"   {category}: 0 reglas")
            
        except Exception as e:
            print(f"❌ Error cargando recursos: {e}")

            for category in ["general", "books", "computers", "cubicles", "biblio"]:
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
            
            # # Boost para categorías específicas (debug)
            # if category != "general":
            #     confidence *= 1.2
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_answer = data["respuesta"]
        
        return best_answer, best_confidence
    
    def process_question(self, question: str, session_id: str = None) -> Dict:
        """
        Procesa pregunta usando texto normalizado con rate limiting
        
        Args:
            question: Pregunta de usuario
            session_id: Asignada por programa principal

        Returns:
            Dict con respuesta y metadatos
    
        Raises:
            ValueError: Si session_id no es proporcionado
        """
        
        if not session_id:
            # Por si acaso, el programa principal siempre deberia proveer una sesion
            if self.debug_mode:
                print(f"❌ ERROR: No se cuenta con un identificador para la sesion")
            
            return {
                "answer": "Error interno: Identificador de sesión no proporcionado. Contacte al administrador.",
                "confidence": 0.0,
                "source": "general",
                "mode": "basic" if not self.use_spacy else "spacy",
                "session_id": None,
                "rate_limited": False,
                "error": "missing_session_id"
            }
        
        # Si no hay identificador (debug)
        identifier = session_id if session_id else "anonimo"

        allowed, wait_time, remaining = self.check_rate_limit(session_id)

        if not allowed:
            if self.debug_mode:
                print(f"⚠️ Se ha excedido el limite para este ID: {identifier}")
            
            return {
                "answer": f"Has realizado demasiadas preguntas. Por favor espera {wait_time} segundos antes de continuar.",
                "confidence": 0.0,
                "source": "general",
                "mode": "basic" if not self.use_spacy else "spacy",
                "session_id": session_id,
                "rate_limited": True,
                "wait_time": wait_time
            }
        
        # Validar entrada
        if not question or not question.strip():
            return {
                "answer": "Haz una pregunta sobre los servicios de la biblioteca.",
                "confidence": 0.0,
                "source": "general",
                "mode": "basic" if not self.use_spacy else "spacy",
                "session_id": session_id,
                "rate_limited": False
            }
        
        session = self.session_manager.get_session(session_id)

        if not session:
            # Redundancia, la sesion se asigna por parte del programa principal
            if self.debug_mode:
                print(f"🆕 Creating new session for: {session_id}")
            
            self.session_manager.create_session_with_id(session_id)
            session = self.session_manager.get_session(session_id)

        # Si esta disponible, contexto se toma en cuenta para mejor categorizacion
        if session.get('last_category'):
            if self.debug_mode:
                print(f"📌 Contexto: última categoría fue '{session['last_category']}'")
        
        # Normalizar la pregunta al inicio 
        question_for_processing = self.matcher.normalize_text(question)
        
        if self.debug_mode:
            print(f"\n{'='*60}")
            print(f" PROCESANDO: '{question}'")
            print(f" Sesión: {session_id[:8]}..." if len(session_id) > 8 else f" Sesión: {session_id}")
            print(f" Normalizado: '{question_for_processing}'")
            if session.get('last_category'):
                print(f" Contexto: última categoría = '{session['last_category']}'")
            print(f"{'='*60}")

        # Paso de contexto para categoria
        category, category_confidence = self.categorize_question(
            question_for_processing,
            context_category=session.get('last_category')
        )
        
        if self.debug_mode:
            print(f" Categoria: {category} (confianza: {category_confidence:.2f})")
        
        # Expandir consulta usando texto normalizado
        if self.use_spacy:
            expanded_queries = self.expand_query_with_spacy(question_for_processing)
        else:
            expanded_queries = self.matcher.expand_with_synonyms(question_for_processing)
        
        if self.debug_mode and expanded_queries:
            print(f" Consultas expandidas ({len(expanded_queries)}): {expanded_queries[:3]}...")
        
        # Buscar en categoria principal
        best_answer, best_confidence = self.search_in_category(category, expanded_queries)
        best_source = category
        
        if self.debug_mode:
            print(f" Mejor coincidencia en {category}: {best_confidence:.3f}")
        
        # Umbrales por categoria
        category_thresholds = {
            "books": 0.4,
            "computers": 0.4,
            "cubicles": 0.4,
            "biblio": 0.35,
            "general": 0.3
        }
        
        threshold = category_thresholds.get(category, 0.4)
        
        # Si no supera umbral, buscar en general
        if best_confidence < threshold:
            if self.debug_mode:
                print(f"⚠️  Confianza baja ({best_confidence:.3f} < {threshold}), probando 'general'")
            
            general_answer, general_confidence = self.search_in_category("general", expanded_queries)
            
            if general_confidence > best_confidence + 0.1:
                best_answer = general_answer
                best_confidence = general_confidence
                best_source = "general"
                
                if self.debug_mode:
                    print(f"🔄 Cambiando a 'general': {general_confidence:.3f}")
        
         # ===== ML CLASSIFIER FALLBACK (Complement to rule-based) =====
        # Only trigger if rule confidence is low AND ML is enabled AND ready
        if (self.enable_ml_classifier and self.ml_ready and 
            best_confidence < self.ml_confidence_threshold):
            
            ml_category, ml_confidence = self._ml_categorize(question)
            
            if self.debug_mode:
                print(f"🤖 ML suggests: {ml_category} (conf: {ml_confidence:.3f})")
            
            # Only use ML suggestion if it's confident AND differs from current best source
            if ml_confidence > self.ml_override_threshold and ml_category != best_source:
                # Search again in ML's suggested category
                ml_answer, ml_match_conf = self.search_in_category(ml_category, expanded_queries)
                
                if ml_match_conf > best_confidence + 0.1:  # Significant improvement
                    best_answer = ml_answer
                    best_confidence = ml_match_conf
                    best_source = ml_category
                    
                    if self.debug_mode:
                        print(f"🔄 ML override: using '{ml_category}' (match conf: {ml_match_conf:.3f})")

        # Usar respuesta de fallback 
        if best_confidence < self.fallback_threshold:
            if self.debug_mode:
                print(f"⚠️  Confianza ({best_confidence:.3f}) por debajo del umbral ({self.fallback_threshold})")
                print("   Activando respuesta 'No sé'")
            
            if self.log_low_confidence:
                self._log_low_confidence(question, session_id, best_confidence, best_source)
            
            best_answer = self.get_idk_response()
            best_confidence = 0.0
            best_source = "fallback"
        
        entities = self.extract_entities(question)

        final_confidence = min(best_confidence * (0.5 + category_confidence * 0.5), 1.0)
        
        result = {
            "answer": best_answer,
            "confidence": round(final_confidence, 3),
            "source": best_source,
            "mode": "basic" if not self.use_spacy else "spacy",
            "entities": entities,
            "session_id": session_id,
            "rate_limited": False,                    
            "rate_limit_remaining": remaining,        
            "details": {
                "category_confidence": round(category_confidence, 3),
                "expanded_queries_count": len(expanded_queries),
                "conversation_count": session.get('conversation_count', 0)
            }
        }
        
        # Guardar en el historial de sesion
        self.session_manager.add_to_history(session_id, question, result)
        
        # Actualizar sesion con el contexto reciente
        self.session_manager.update_session(session_id, {
            'last_category': best_source,
            'last_entities': entities,
            'last_question': question,
            'last_response': best_answer[:200]
        })
        
        if self.debug_mode:
            print(f"\n RESULTADO FINAL:")
            print(f"   Respuesta: {best_answer[:80]}...")
            print(f"   Confianza: {final_confidence:.3f}")
            print(f"   Conversación #{session.get('conversation_count', 0)}")
            print(f"   Rate limit restante: {remaining}/{self.rate_limiter.max_requests}")
            print(f"{'='*60}\n")
        
        return result
    
    def _train_classifier(self):
        """
        Train a logistic regression classifier using the questions from the knowledge base.
        The classifier learns to predict the category (books, computers, etc.) from the text.
        """
        try:
            texts = []
            labels = []
            
            # Extract all questions and their categories from the knowledge base
            for category, knowledge in self.knowledge_base.knowledge.items():
                if not knowledge:
                    continue
                for rule_id, rule_data in knowledge.items():
                    for pregunta in rule_data.get("preguntas", []):
                        # Normalize the question before training
                        normalized = self.matcher.normalize_text(pregunta)
                        if normalized and len(normalized) > 3:
                            texts.append(normalized)
                            labels.append(category)
            
            if len(texts) < 10:
                print(f"⚠️  ML Classifier: Insufficient training data ({len(texts)} examples). Disabling.")
                self.ml_ready = False
                return
            
            # Create TF-IDF vectorizer (unigrams + bigrams)
            self.vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=800,
                sublinear_tf=True
            )
            X = self.vectorizer.fit_transform(texts)
            
            # Train logistic regression classifier
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            self.classifier.fit(X, labels)
            
            self.ml_ready = True
            print(f"✅ ML Classifier trained on {len(texts)} examples across {len(set(labels))} categories")
            
            if self.debug_mode:
                # Print top features per category (optional)
                self._print_top_features()
                
        except Exception as e:
            print(f"❌ ML Classifier training failed: {e}")
            self.ml_ready = False
    
    def _ml_categorize(self, question: str) -> Tuple[str, float]:
        """
        Use the trained ML classifier to predict the category of a question.
        Returns (category, confidence) where confidence is the predicted probability.
        """
        if not self.ml_ready or self.classifier is None or self.vectorizer is None:
            return "general", 0.0
        
        try:
            normalized = self.matcher.normalize_text(question)
            X = self.vectorizer.transform([normalized])
            
            predicted_category = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            confidence = max(probabilities)
            
            return predicted_category, confidence
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️  ML prediction error: {e}")
            return "general", 0.0
    
    def _print_top_features(self, n: int = 5):
        """Print the top n features (words/bigrams) for each category."""
        if not self.ml_ready:
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        for i, category in enumerate(self.classifier.classes_):
            coef = self.classifier.coef_[i]
            top_indices = coef.argsort()[-n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            print(f"   {category}: {', '.join(top_features)}")

    def get_fallback_response(self, category: str, question: str = "") -> str:
        """Respuesta de fallback especifica por categoría"""
        
        question_lower = question.lower()
        
        if "libro" in question_lower or "texto" in question_lower:
            return "Para préstamos de libros, presenta tu carné en recepción. Se permiten hasta 3 libros por 15 días."

        if "cubiculo" in question_lower or "sala" in question_lower or "cabina" in question_lower:
            return "Para reservar cubículos, acércate a la recepción con tu carné. Se permite 1 reserva por día de máximo 3 horas."
        
        if "computadora" in question_lower or "ordenador" in question_lower:
            return "Las computadoras están disponibles por orden de llegada. Máximo 2 horas de uso. Presenta tu carné."
        
        if any(word in question_lower for word in ["biblio", "asistente", "quién eres", "quien eres", "qué haces", "que haces"]):
            return "¡Hola! Soy Biblio, tu asistente virtual. Puedo ayudarte con preguntas sobre libros, computadoras, cubículos y servicios de la biblioteca. ¿Qué necesitas saber?"

        # Respuestas genericas por categoria
        fallback_responses = {
            "books": "Información sobre libros disponible en recepción. Horario de atención: 8 AM a 6 PM.",
            "computers": "Consulta las reglas de uso de computadoras en el área de tecnología.",
            "cubicles": "Para reservar cubículos, visita la recepción con identificación.",
            "general": "No he entendido la pregunta, ¿podrías reformularla?.",
            "biblio": "Soy Biblio, tu asistente. Pregúntame sobre libros, computadoras, cubículos o servicios generales."
        }
        
        return fallback_responses.get(category, fallback_responses["general"])
    
    def get_idk_response(self) -> str:
        """Basicamente "No se" como repsuesta"""
        return (
            "Lo siento, no tengo información suficiente para responder a esa pregunta. "
            "¿Podrías reformularla o consultar directamente en el mostrador de atención al usuario? "
            "¡Estaré encantado de ayudarte con otras dudas sobre la biblioteca!"
        )

    def _log_low_confidence(self, question: str, session_id: str, confidence: float, attempted_source: str):
        """Log a low-confidence query para retroalimentacion."""
        import json
        from datetime import datetime
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "confidence": round(confidence, 4),
            "attempted_source": attempted_source
        }
        
        try:
            with open(self.low_confidence_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Error writing low-confidence log: {e}")
    
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
        print(" Creando chatbot con Spacy...")
    else:
        print(" Creando chatbot en modo básico...")
        if SPACY_AVAILABLE and force_basic:
            print(" Nota: Spacy está disponible pero se forzó modo básico")
    
    return ChatBot(use_spacy=use_spacy)

def diagnose_spacy():
    """Diagnóstico de Spacy (problema de dependencia?)"""
    print("\n" + "="*60)
    print(" DIAGNÓSTICO DE SPACY")
    print("="*60)
    
    if not SPACY_AVAILABLE:
        print("❌ Spacy no está instalado")
        return False
    
    print("✅ Spacy está instalado")
    
    try:
        nlp = spacy.load("es_core_news_sm")
        print("✅ Modelo 'es_core_news_sm' cargado correctamente")
        print(f" Pipeline: {nlp.pipe_names}")
        print(f" Vocabulario: {len(nlp.vocab)} palabras")
        
        # Probar procesamiento
        test_text = "reservar un cubículo"
        doc = nlp(test_text)
        print(f"\n Prueba con '{test_text}':")
        for token in doc:
            print(f"   '{token.text}' → Lemma: '{token.lemma_}', POS: '{token.pos_}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        # Solución: python -m spacy download es_core_news_sm"
        return False

if __name__ == "__main__":
    # Ejecutar diagnóstico
    diagnose_spacy()
    
    # Crear y probar chatbot
    print(f"\n{'='*60}")
    print("PRUEBA DEL CHATBOT")
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
        print(f"    📍 {response['source']} | {response['mode']} | {response['confidence']}")