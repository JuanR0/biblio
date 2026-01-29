from typing import Dict, List, Set
import json
from pathlib import Path

class SynonymManager:
    def __init__(self):
        self.synonym_groups: Dict[str, Set[str]] = {}
        self.word_to_group: Dict[str, str] = {}
        self._load_synonyms()
    
    def _load_synonyms(self):
        """Carga sinónimos desde archivo JSON"""
        synonyms_path = Path("data/knowledge/synonyms.json")
        
        if synonyms_path.exists():
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                synonym_groups = json.load(f)
                
            for group_name, words in synonym_groups.items():
                normalized_words = {self._normalize_word(w) for w in words}
                self.synonym_groups[group_name] = normalized_words
                
                # Mapeo inverso palabra -> grupo
                for word in normalized_words:
                    self.word_to_group[word] = group_name
        else:
            # Sinónimos básicos por defecto
            self._create_default_synonyms()
    
    def _create_default_synonyms(self):
        """Crea sinónimos básicos por defecto"""
        default_synonyms = {
            "prestar": ["pedir prestado", "tomar prestado", "retirar", "sacar"],
            "libro": ["libros", "ejemplar", "volumen", "obra"],
            "devolver": ["entregar", "regresar", "restituir"],
            "multa": ["sanción", "penalización", "recargo"],
            "renovar": ["prorrogar", "extender", "prolongar"],
            "horario": ["horas", "horas de atención", "tiempo", "jornada"],
            "abierto": ["abrir", "abierto al público", "atender"],
            "cerrado": ["cerrar", "no abierto", "sin atención"]
        }
        
        for group_name, words in default_synonyms.items():
            normalized_words = {self._normalize_word(w) for w in words + [group_name]}
            self.synonym_groups[group_name] = normalized_words
            
            for word in normalized_words:
                self.word_to_group[word] = group_name
    
    def _normalize_word(self, word: str) -> str:
        """Normaliza una palabra para comparación"""
        return word.lower().strip()
    
    def get_synonyms(self, word: str) -> Set[str]:
        """Obtiene todos los sinónimos de una palabra"""
        normalized = self._normalize_word(word)
        
        if normalized in self.word_to_group:
            group_name = self.word_to_group[normalized]
            return self.synonym_groups[group_name]
        
        return {normalized}
    
    def is_synonym(self, word1: str, word2: str) -> bool:
        """Verifica si dos palabras son sinónimas"""
        norm1 = self._normalize_word(word1)
        norm2 = self._normalize_word(word2)
        
        if norm1 == norm2:
            return True
        
        group1 = self.word_to_group.get(norm1)
        group2 = self.word_to_group.get(norm2)
        
        return group1 is not None and group1 == group2
    
    def expand_with_synonyms(self, words: List[str]) -> Set[str]:
        """Expande una lista de palabras con sus sinónimos"""
        expanded = set()
        
        for word in words:
            expanded.update(self.get_synonyms(word))
        
        return expanded
    
    def find_keyword_synonyms(self, question: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Encuentra qué sinónimos de keywords aparecen en la pregunta"""
        found_synonyms = {}
        
        for keyword in keywords:
            keyword_synonyms = self.get_synonyms(keyword)
            
            # Buscar cada sinónimo en la pregunta
            found = []
            for synonym in keyword_synonyms:
                if synonym in question.lower():
                    found.append(synonym)
            
            if found:
                found_synonyms[keyword] = found
        
        return found_synonyms