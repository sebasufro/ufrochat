# providers/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time

class LLMProvider(ABC):
    """
    Protocolo base (Adapter) para todos los proveedores de modelos de lenguaje (LLM).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Retorna el nombre del proveedor (ej. 'ChatGPT', 'DeepSeek')."""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Envía un historial de mensajes al modelo y retorna la respuesta.
        
        Args:
            messages: Lista de mensajes en formato OpenAI (rol, contenido).
            **kwargs: Parámetros adicionales (ej. temperatura, top_p, etc.).
            
        Returns:
            Un diccionario con la respuesta y metadatos (ej. tokens usados, costo).
        """
        pass

    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """
        Estima el costo en dólares USD basado en el uso de tokens.
        Debe ser implementado en cada subclase.
        """
        return 0.0

    def __repr__(self) -> str:
        return f"<{self.name} Provider>"
