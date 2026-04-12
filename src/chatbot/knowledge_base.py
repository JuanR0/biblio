import json
import os
from typing import Dict, Any, Optional

class KnowledgeBase:
    def __init__(self, knowledge_path: str):
        self.knowledge_path = knowledge_path
        self.knowledge = {
            "general": None,
            "books": None,
            "computers": None,
            "cubicles": None
        }
    
    def load_knowledge_file(self, filename: str) -> Optional[Dict]:
        """Carga un archivo JSON de conocimiento con manejo de errores"""
        filepath = os.path.join(self.knowledge_path, filename)
        
        # Asegurarse de que el archivo existe
        if not os.path.exists(filepath):
            print(f"❌ Archivo no encontrado: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Solo acepta archivos JSON 
            if not isinstance(content, dict):
                print(f"❌ Estructura inválida en {filename}: debe ser un objeto JSON")
                return None
            
            print(f"✅ Archivo cargado: {filename}")
            return content
            
        except json.JSONDecodeError as e:
            print(f"❌ Error de JSON en {filename}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error leyendo {filename}: {e}")
            return None
    
    def load_all_knowledge(self):
        """Carga todos los archivos de conocimiento"""
        file_mapping = {
            "general": "general_rules.json",
            "books": "books_rules.json",
            "computers": "computers_rules.json",
            "cubicles": "cubicles_rules.json",
            "biblio": "biblio_rules.json"
        }
        
        for category, filename in file_mapping.items():
            print(f"\n📂 Cargando {category} desde {filename}...")
            knowledge_data = self.load_knowledge_file(filename)
            
            if knowledge_data:
                self.knowledge[category] = knowledge_data
                print(f"   ✅ {len(knowledge_data)} reglas cargadas para {category}")
            else:
                print(f"   ⚠️  Las relgas no han sido encontradas {category}")
                
                self.knowledge[category] = {}
                
                # Datos de ejemplo como respaldo
                if category in ["computers", "cubicles"]:
                    print(f"   📝 Creando datos de ejemplo para {category}...")
                    self.knowledge[category] = self._create_example_data(category)
    
    def _create_example_data(self, category: str) -> Dict:
        """Datos de ejemplo en caso de no encontrar los archivos de conocimiento"""
        if category == "computers":
            return {
                "uso_computadoras": {
                    "preguntas": ["uso de computadoras", "cómo usar ordenador"],
                    "respuesta": "Las computadoras están disponibles por orden de llegada. Máximo 2 horas de uso."
                }
            }
        elif category == "cubicles":
            return {
                "reserva_cubiculos": {
                    "preguntas": ["cómo reservar cubículo", "sala de estudio"],
                    "respuesta": "Reserva cubículos en recepción. Máximo 3 horas por día."
                }
            }
        elif category == "biblio":
            return {
                "presentacion_biblio": {
                    "preguntas": ["quién eres", "qué haces", "presentación"],
                    "respuesta": "Soy Biblio, tu asistente virtual de la biblioteca. Puedo ayudarte con preguntas sobre libros, computadoras, cubículos y servicios generales."
                }
            }
        return {}
    
    def get_knowledge(self, category: str) -> Dict:
        """Obtiene conocimiento de una categoría especifica"""
        return self.knowledge.get(category, {})
    
    def list_loaded_categories(self):
        """Lista las categorias cargadas y sus reglas"""
        print("\n" + "="*50)
        print("CATEGORIAS CARGADAS")
        print("="*50)
        
        for category, data in self.knowledge.items():
            if data:
                print(f"\n📁 {category.upper()}:")
                for key, rule in data.items():
                    preguntas = rule.get("preguntas", [])
                    print(f"   ├─ {key}: {len(preguntas)} preguntas")
                    print(f"   └─ Ejemplo: {preguntas[0] if preguntas else 'Sin preguntas'}")
            else:
                print(f"\n📁 {category.upper()}: VACIO O NO CARGADO")