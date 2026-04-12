"""
Rate Limiter para asistente virtual 
"""

import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime


class RateLimiter:
    """Rate limiter en memoria."""
    
    def __init__(self, max_requests: int = 2, window_seconds: int = 10):
        """
        Inicializar rate limiter.
        
        Args:
            max_requests: Limite de solicitudes por cada intervalo
            window_seconds: Duracion de intervalo (segundos)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Solicitudes por id
        # Formato: {id: [timestamp1, timestamp2, ...]}
        self.requests: Dict[str, List[float]] = defaultdict(list)
        
        self.blocked_requests: Dict[str, int] = defaultdict(int)
        
        # Limpieza
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  
    
    def is_allowed(self, identifier: str) -> Tuple[bool, int]:
        """
        Validar si se permite una solicitud del ID.
        
        Args:
            ID
            
        Returns:
            (is_allowed, wait_time_or_remaining)
            - Allowed: (True, remaining_requests)
            - Blocked: (False, seconds_to_wait)
        """
        now = time.time()
        
        # Limpieza de solicitudes viejas
        self._cleanup_if_needed(now)
        
        # Solicitar historial id
        timestamps = self.requests[identifier]
        
        # Limpiar solicitudes mas viejas que el intervalo
        while timestamps and timestamps[0] < now - self.window_seconds:
            timestamps.pop(0)
        
        # Validar si se excedio el limite
        if len(timestamps) >= self.max_requests:
            # Cuanto falta para que expire ultimo request, y solicitudes bloqueadas 
            oldest = timestamps[0]
            wait_time = int(self.window_seconds - (now - oldest)) + 1
            
            self.blocked_requests[identifier] += 1
            
            return False, wait_time

        timestamps.append(now)
        remaining = self.max_requests - len(timestamps)
        
        return True, remaining
    
    def get_remaining(self, identifier: str) -> int:
        """Recibe requests restantes."""
        now = time.time()
        timestamps = self.requests.get(identifier, [])
        
        while timestamps and timestamps[0] < now - self.window_seconds:
            timestamps.pop(0)
        
        return self.max_requests - len(timestamps)
    
    def get_reset_time(self, identifier: str) -> Optional[int]:
        """Regresa los segundos restantes para un nuevo in tervalo de preguntas, solo si se ha excedido."""
        now = time.time()
        timestamps = self.requests.get(identifier, [])
        
        if len(timestamps) < self.max_requests:
            return None
        
        oldest = timestamps[0]
        return int(self.window_seconds - (now - oldest)) + 1
    
    def reset(self, identifier: Optional[str] = None):
        """
        Resetea solicitudes e intervalo de preguntas para ID/Todos
        
        Args:
            identifier: Si se recibe ID, se resetea solo este.
                        Si no, todos son reseteados.
        """
        if identifier:
            self.requests.pop(identifier, None)
            self.blocked_requests.pop(identifier, None)
        else:
            self.requests.clear()
            self.blocked_requests.clear()
    
    def get_stats(self) -> Dict:
        """Estadisticas de rate limiter."""
        total_active = len(self.requests)
        total_blocked = sum(self.blocked_requests.values())
        
        # Top bloqueos
        top_blocked = sorted(
            self.blocked_requests.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "active_identifiers": total_active,
            "total_blocked_requests": total_blocked,
            "max_requests_per_window": self.max_requests,
            "window_seconds": self.window_seconds,
            "top_blocked": [{"identifier": id, "blocked": count} for id, count in top_blocked]
        }
    
    def _cleanup_if_needed(self, now: float):
        """Limpieza de solicitudes viejas para liberar memoria"""
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        
        # Limpiar IDs sin actividad reciente
        cutoff = now - (self.window_seconds * 2)  # 2 intervalos para solicitudes intermedias
        to_remove = []
        
        for identifier, timestamps in self.requests.items():
            # Remover aquellos sin solicitudes
            if not timestamps or timestamps[-1] < cutoff:
                to_remove.append(identifier)
        
        for identifier in to_remove:
            del self.requests[identifier]
            self.blocked_requests.pop(identifier, None)


class TieredRateLimiter:
    """
    Rate limitter por tipo de usuario.
    
    Ej.:
        limiter = TieredRateLimiter({
            'comun': (2, 10),     
            'admin': (100, 60)    
        })
    """
    
    def __init__(self, tiers: Dict[str, Tuple[int, int]]):
        """
        Initialize tiered rate limiter.
        
        Args:
            tiers: Dictionary mapping tier names (max_requests, window_seconds)
        """
        self.limiters = {}
        for tier, (max_req, window) in tiers.items():
            self.limiters[tier] = RateLimiter(max_req, window)
    
    def is_allowed(self, identifier: str, tier: str = "comun") -> Tuple[bool, int]:
        """Validar ID y tier para rate limitter."""
        limiter = self.limiters.get(tier, self.limiters["comun"])
        return limiter.is_allowed(identifier)
    
    def get_remaining(self, identifier: str, tier: str = "comun") -> int:
        """Validar el limite para tier en especifico."""
        limiter = self.limiters.get(tier, self.limiters["comun"])
        return limiter.get_remaining(identifier)
    
    def reset(self, identifier: Optional[str] = None, tier: Optional[str] = None):
        """Reiniciar rate limitter para ID."""
        if tier:
            self.limiters[tier].reset(identifier)
        else:
            for limiter in self.limiters.values():
                limiter.reset(identifier)