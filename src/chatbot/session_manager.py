import uuid
import time
import json
from typing import Dict, Optional, Any
from datetime import datetime


class SessionManager:
    """Manejador de sesiones mediante dictionaries"""
    
    def __init__(self, session_timeout: int = 600):
        """
        Args:
            session_timeout: Session timeout en segundos (default 10 minutes)
        """
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = session_timeout
    
    def create_session_with_id(self, session_id: str, initial_data: Optional[Dict] = None) -> bool:
        """Crear sesion con ID asignado por programa principal"""
        if session_id in self.sessions:
        # Solo actualizar timestamp
            self.sessions[session_id]['last_activity'] = time.time()
            return True
        
        self.sessions[session_id] = {
            'created_at': time.time(),
            'last_activity': time.time(),
            'history': [],
            'last_category': None,
            'last_entities': {},
            'conversation_count': 0,
            'data': initial_data or {}
        }
        return True

    def create_session(self, initial_data: Optional[Dict] = None) -> str:
        """Creacion de sesion con ID (provisional)"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'created_at': time.time(),
            'last_activity': time.time(),
            'history': [],
            'last_category': None,
            'last_entities': {},
            'conversation_count': 0,
            'data': initial_data or {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Obtener informacion de sesion activa si existe"""
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if time.time() - session['last_activity'] > self.session_timeout:
            del self.sessions[session_id]
            return None

        session['last_activity'] = time.time()
        return session
    
    def update_session(self, session_id: str, data: Dict) -> bool:
        """Actualizar informacion de sesion"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.update(data)
        session['last_activity'] = time.time()
        return True
    
    def add_to_history(self, session_id: str, question: str, response: Dict) -> bool:
        """Agregar pregunta-respuesta al historail de la sesion"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['history'].append({
            'question': question,
            'response': response['answer'][:200],  
            'category': response['source'],
            'timestamp': time.time()
        })
        
        # limite de categorias para preservar memoria
        if len(session['history']) > 10:
            session['history'] = session['history'][-10:]
        
        session['conversation_count'] += 1
        session['last_category'] = response['source']
        session['last_entities'] = response.get('entities', {})
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Eliminar sesiones especificas"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def delete_all_sessions(self) -> int:
        """Eliminar todas las sesiones"""
        count = len(self.sessions)
        self.sessions.clear()
        return count
    
    def get_active_sessions_count(self) -> int:
        """Limpieza de sesiones expiradas para obtener cuenta de activas"""
        # limpiar sesiones expiradas
        expired = []
        for sid, session in self.sessions.items():
            if time.time() - session['last_activity'] > self.session_timeout:
                expired.append(sid)
        
        for sid in expired:
            del self.sessions[sid]
        
        return len(self.sessions)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Info de sesion"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            'session_id': session_id,
            'conversation_count': session.get('conversation_count', 0),
            'last_category': session.get('last_category'),
            'last_activity': session.get('last_activity'),
            'created_at': session.get('created_at')
        }