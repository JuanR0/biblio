import json
import os
from typing import List, Dict, Set
import re
from difflib import SequenceMatcher

class QueryMatcher:
    def __init__(self, synonyms_path: str):
        self.synonyms_path = synonyms_path
        self.synonyms = {}
        
        self.stop_words = {
            'cómo', 'cuál', 'dónde', 'qué', 'cuánto', 'cuánta', 'cuántos', 'cuántas',
            'para', 'por', 'con', 'sin', 'sobre', 'bajo', 'entre', 'hacia', 'desde',
            'se', 'un', 'una', 'unos', 'unas', 'el', 'la', 'los', 'las', 'lo',
            'de', 'del', 'al', 'y', 'o', 'pero', 'a', 'en', 'que', 'es', 'son',
            'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
            'me', 'te', 'nos', 'os', 'le', 'les', 'mi', 'tu', 'su', 'nuestro',
            'vuestro', 'sus', 'muy', 'mas', 'más', 'menos', 'tan', 'tanto',
            'como', 'cuando', 'donde', 'mientras', 'aunque', 'porque', 'si',
            'sí', 'no', 'también', 'además', 'entonces', 'luego', 'ahora',
            'antes', 'después', 'siempre', 'nunca', 'a veces', 'quizás',
            'conseguir', 'obtener', 'tener', 'hacer', 'usar', 'utilizar',
            'necesitar', 'querer', 'poder', 'deber', 'saber', 'conocer'
        }
        
        # Palabras mutuamente excluyentes, no pueden ser sinonimas entre categorias
        self.exclusive_words = {
            "libro": ["computadora", "cubiculo", "equipo", "sala"],
            "computadora": ["libro", "cubiculo", "texto", "obra"],
            "cubiculo": ["libro", "computadora", "texto", "equipo"]
        }
        
        self.debug = False
    
    def load_synonyms(self):
        """Carga el archivo de sinónimos con validación"""
        try:
            filepath = os.path.join(self.synonyms_path, "synonyms.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                self.synonyms = json.load(f)
            
            self.clean_synonyms()
            
        except Exception as e:
            print(f"❌ Error cargando sinónimos: {e}")
            self.synonyms = {}
    
    def clean_synonyms(self):
        """Limpia sinonimos que puedan causar confusión entre categorías"""
        # NO permitir que 'conseguir' sea sinónimo de 'prestar' para libros
        if "conseguir" in self.synonyms and "prestar" in self.synonyms["conseguir"]:
            self.synonyms["conseguir"].remove("prestar")
    
    def normalize_text(self, text: str) -> str:
        """Normaliza el texto manteniendo palabras clave importantes"""
        if not text:
            return ""
        
        text = text.lower()
        
        # Mantener palabras clave importantes sin modificar
        important_words = ["cubiculo", "libro", "computadora", "reserva", "prestamo"]
        
        # Reemplazar sinónimos problematicos 
        problematic_synonyms = {
            "conseguir": "reservar",
            "obtener": "reservar",
            "tomar": "prestar"
        }
        
        for problematic, replacement in problematic_synonyms.items():
            if problematic in text:
                text = text.replace(problematic, replacement)
        
        # Normalizar acentos
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ü': 'u', 'ñ': 'n'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Quitar caracteres especiales
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """Expande la consulta incluyendo sinonimos"""
        normalized = self.normalize_text(query)
        
        if not normalized:
            return [""]
        
        words = normalized.split()
        expanded_queries = [normalized]
        
        # En caso de palabra clave de categoria, no expandir
        category_keywords = ["cubiculo", "libro", "computadora"]
        
        for i, word in enumerate(words):
            # En caso de palabra de categoria, no expandir con sinonimos
            if word in category_keywords:
                continue
                
            # Expandir solo verbos/acciones comunes
            if word in self.synonyms and len(word) > 2:
                for synonym in self.synonyms[word][:1]:  # Solo 1 sinónimo
                    new_words = words.copy()
                    new_words[i] = synonym
                    expanded_queries.append(' '.join(new_words))
        
        # Agregar versión sin stop words, manteniendo palabras clave
        important_words = []
        for word in words:
            if word not in self.stop_words or word in category_keywords:
                important_words.append(word)
        
        if important_words and ' '.join(important_words) != normalized:
            expanded_queries.append(' '.join(important_words))
        
        # Eliminar duplicados
        unique_queries = []
        for q in expanded_queries:
            if q not in unique_queries:
                unique_queries.append(q)
        
        if self.debug:
            print(f"   Consultas generadas: {unique_queries}")
        
        return unique_queries
    
    def calculate_similarity(self, queries: List[str], target_phrases: List[str]) -> float:
        """Calcula similitud con penalizacion por categorías cruzadas"""
        if not queries or not target_phrases:
            return 0.0
        
        max_similarity = 0.0
        
        for query in queries:
            if not query:
                continue
                
            query_normalized = self.normalize_text(query)
            query_words = set(query_normalized.split())
            
            for phrase in target_phrases:
                if not phrase:
                    continue
                    
                phrase_normalized = self.normalize_text(phrase)
                phrase_words = set(phrase_normalized.split())
                
                # Comprobacion de palabras criticas
                critical_words_query = query_words & {"cubiculo", "libro", "computadora"}
                critical_words_phrase = phrase_words & {"cubiculo", "libro", "computadora"}
                
                # Penalizacion en ausencia de coincidencias para palabras criticas
                if critical_words_query and critical_words_phrase:
                    if critical_words_query != critical_words_phrase:
                  
                        continue  
                
                # Similitud Jaccard 
                query_content = query_words - self.stop_words
                phrase_content = phrase_words - self.stop_words
                
                keyword_similarity = 0.0
                if query_content and phrase_content:
                    intersection = len(query_content & phrase_content)
                    union = len(query_content | phrase_content)
                    
                    if union > 0:
                        keyword_similarity = intersection / union
                
                # Similitud textual
                textual_similarity = SequenceMatcher(
                    None, 
                    query_normalized, 
                    phrase_normalized
                ).ratio()
                
                # Bonus por coincidencia de verbos importantes
                action_verbs = {"reservar", "prestar", "usar", "devolver"}
                common_actions = query_content & phrase_content & action_verbs
                bonus = len(common_actions) * 0.05
                
                # Penalización por no contener en la frase palabras clave
                if critical_words_query and not (critical_words_query & phrase_words):
                    bonus -= 0.3  
                
                # Combinar
                combined = (keyword_similarity * 0.7) + (textual_similarity * 0.2) + bonus
                combined = max(0.0, combined)  # No negativa
                
                if combined > max_similarity:
                    max_similarity = combined
        
        return min(max_similarity, 1.0)