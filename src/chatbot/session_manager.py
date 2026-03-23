import uuid
import time
import json
from typing import Dict, Optional, Any
from datetime import datetime


class SessionManager:
    """Simple session manager using Python dictionaries"""
    
    def __init__(self, session_timeout: int = 1800):
        """
        Args:
            session_timeout: Session timeout in seconds (default 30 minutes)
        """
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = session_timeout
    
    def create_session_with_id(self, session_id: str, initial_data: Optional[Dict] = None) -> bool:
        """Create a session with a specific ID (not UUID)"""
        if session_id in self.sessions:
            return False  
        
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
        """Create a new session and return session ID"""
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
        """Get session data if exists and not expired"""
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if time.time() - session['last_activity'] > self.session_timeout:
            del self.sessions[session_id]
            return None

        session['last_activity'] = time.time()
        return session
    
    def update_session(self, session_id: str, data: Dict) -> bool:
        """Update session data"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.update(data)
        session['last_activity'] = time.time()
        return True
    
    def add_to_history(self, session_id: str, question: str, response: Dict) -> bool:
        """Add question-answer pair to session history"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['history'].append({
            'question': question,
            'response': response['answer'][:200],  
            'category': response['source'],
            'timestamp': time.time()
        })
        
        # limitatrse a 10 categorias para cuidar memoria
        if len(session['history']) > 10:
            session['history'] = session['history'][-10:]
        
        session['conversation_count'] += 1
        session['last_category'] = response['source']
        session['last_entities'] = response.get('entities', {})
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def delete_all_sessions(self) -> int:
        """Delete all sessions (useful for testing)"""
        count = len(self.sessions)
        self.sessions.clear()
        return count
    
    def get_active_sessions_count(self) -> int:
        """Get number of active sessions (cleans expired first)"""
        # limpiar sesiones expiradas
        expired = []
        for sid, session in self.sessions.items():
            if time.time() - session['last_activity'] > self.session_timeout:
                expired.append(sid)
        
        for sid in expired:
            del self.sessions[sid]
        
        return len(self.sessions)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get summary of a session (without full history)"""
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