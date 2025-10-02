# providers/chatgpt.py

import os
from openai import OpenAI
from typing import List, Dict, Any
from .base import LLMProvider
import time

class ChatGPTProvider(LLMProvider):
    """Adaptador para el API de OpenAI (GPT-3.5/GPT-4)."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        # La clave API se lee automáticamente desde la variable de entorno OPENAI_API_KEY
        self.client = OpenAI()
        self.model = model
        self._name = f"ChatGPT ({model})"

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Envía los mensajes al endpoint de Chat Completions de OpenAI."""

        start_time = time.time()

        # Parámetros por defecto para robustez
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1024),
            **kwargs
        }

        response = self.client.chat.completions.create(**params)
        end_time = time.time()

        # Extracción de la información relevante
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
        """Estima el costo de la consulta con GPT-3.5-Turbo."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        prompt_cost = (prompt_tokens / 1000) * PRICE_PER_1K_PROMPT
        completion_cost = (completion_tokens / 1000) * PRICE_PER_1K_COMPLETION

        return prompt_cost + completion_cost
