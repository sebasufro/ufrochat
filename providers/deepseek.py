# providers/deepseek.py

import os
from openai import OpenAI
from typing import List, Dict, Any
from .base import LLMProvider
import time

class DeepSeekProvider(LLMProvider):
    """Adaptador para DeepSeek, compatible con el API de OpenAI."""
    
    PRICE_PER_1K_PROMPT_DS = 0.00025   # Precio por 1,000 tokens de entrada (DeepSeek Chat)
    PRICE_PER_1K_COMPLETION_DS = 0.0010  # Precio por 1,000 tokens de salida (DeepSeek Chat)

    def __init__(self, model: str = "deepseek-chat"):
        # DeepSeek usa la misma estructura de cliente OpenAI pero con base_url
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1" 
        )
        self.model = model
        self._name = f"DeepSeek ({model})"

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Envía los mensajes usando el cliente OpenAI-compatible."""

        start_time = time.time()
        
        # ... (código para llamar a la API) ...
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1024),
            **kwargs
        }

        response = self.client.chat.completions.create(**params)
        end_time = time.time()

        choice = response.choices[0]
        usage = response.usage.model_dump()

        # Estimar el costo antes de retornar
        cost = self.estimate_cost(usage)

        return {
            "text": choice.message.content,
            "usage": response.usage.model_dump(),
            "model": self.model,
            "finish_reason": choice.finish_reason,
            "latency_sec": end_time - start_time,
            "estimated_cost_usd": cost
        }

    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """Estima el costo de la consulta con DeepSeek-Chat."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Acceso correcto a las constantes de clase
        prompt_cost = (prompt_tokens / 1000) * self.PRICE_PER_1K_PROMPT_DS
        completion_cost = (completion_tokens / 1000) * self.PRICE_PER_1K_COMPLETION_DS

        return prompt_cost + completion_cost