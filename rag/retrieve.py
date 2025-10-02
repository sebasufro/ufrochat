# rag/retrieve.py

import pandas as pd
import numpy as np
import faiss
import os
import time

from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

from providers.base import LLMProvider
from rag.prompts import SYSTEM_PROMPT, REWRITE_QUERY_PROMPT

# --- 5.2.1 Clase RAGManager ---

class RAGManager:
    """Orquestador del pipeline Retrieval-Augmented Generation (RAG)."""

    def __init__(self, index_dir: str, embedding_model_name: str):
        # Cargar modelo de embeddings para consultas
        self.embed_model = SentenceTransformer(embedding_model_name)

        # Cargar índice FAISS y metadatos
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.chunks_path = os.path.join(index_dir, "chunks.parquet")

        print(f"Cargando índice FAISS desde {self.index_path}...")
        self.index = faiss.read_index(self.index_path)

        print(f"Cargando metadatos de chunks desde {self.chunks_path}...")
        self.chunks_df = pd.read_parquet(self.chunks_path)

    def _embed_query(self, query: str) -> np.ndarray:
        """Convierte una consulta de texto a vector."""
        # Se asegura de usar float32 para la compatibilidad con FAISS
        return self.embed_model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )[0].astype('float32')

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Dict[str, any]], List[float]]:
        """
        Realiza la búsqueda vectorial en FAISS.
        Retorna: Lista de chunks (con metadatos y texto) y sus scores.
        """
        start_time = time.time()

        qv = self._embed_query(query)

        # D: distancias (scores), I: índices de los vectores
        D, I = self.index.search(np.array([qv]), k)

        end_time = time.time()
        retrieval_latency = end_time - start_time

        results = []
        scores = D[0].tolist()

        # Mapear los índices I a los chunks del DataFrame
        for i, score in zip(I[0], scores):
            # Obtener el chunk por su índice de fila en el DataFrame
            chunk_data = self.chunks_df.iloc[i].to_dict()
            results.append(chunk_data)

        return results, scores, retrieval_latency

    def _format_context(self, retrieved_chunks: List[Dict[str, any]]) -> str:
        """Formatea los chunks recuperados en un string de contexto para el LLM."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Formato de la fuente para que el LLM lo use en la cita [Título, URL]
            source_info = f"[{chunk['doc_title']}, {chunk['doc_url']}]"

            context_parts.append(
                f"--- FUENTE {i} ({source_info}) ---\n"
                f"{chunk['text']}\n"
            )
        return "\n\n".join(context_parts)

    def _rewrite_query(self, provider: LLMProvider, query: str) -> str:
        """Re-escribe la consulta para mejor precisión (Opcional)."""
        messages = [
            {"role": "system", "content": "Actúa como un optimizador de consultas."},
            {"role": "user", "content": REWRITE_QUERY_PROMPT.format(query=query)}
        ]

        try:
            response = provider.chat(messages, temperature=0.0)
            return response['text'].strip()
        except Exception as e:
            print(f"Advertencia: Falló la re-escritura de consulta con {provider.name}. Usando consulta original. Error: {e}")
            return query

    # --- 5.2.2 Función de Síntesis (Core RAG) ---

    def synthesize(self, provider: LLMProvider, original_query: str, k: int = 5, use_rewrite: bool = False) -> Dict[str, any]:
        """
        Ejecuta el pipeline RAG (opcional: rewrite -> retrieve -> synthesize).
        """

        # 1. Re-escritura de Consulta (Opcional) [cite: 15]
        effective_query = original_query
        if use_rewrite:
            effective_query = self._rewrite_query(provider, original_query)

        # 2. Recuperación top-k [cite: 15]
        retrieved_chunks, scores, retrieval_latency = self.retrieve(effective_query, k=k)

        # 3. Formato del Contexto
        context = self._format_context(retrieved_chunks)

        # 4. Preparar mensajes con el Prompt de Sistema y Contexto [cite: 48]
        system_content = SYSTEM_PROMPT.format(context=context)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": original_query}
        ]

        # 5. Síntesis (Generación de Respuesta) [cite: 15]
        llm_response = provider.chat(messages)

        total_latency = retrieval_latency + llm_response['latency_sec']

        # 6. Post-proceso y resultados finales
	# Aquí se podría implementar una verificación de la política de abstención.

        return {
            "answer": llm_response['text'],
            "provider": provider.name,
            "original_query": original_query,
            "effective_query": effective_query,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_scores": scores,
            "retrieval_latency_sec": retrieval_latency, # Latencia de Recuperación
            "llm_latency_sec": llm_response['latency_sec'], # Latencia LLM
            "total_latency_sec": total_latency, # Latencia Total (End-to-end)
            "estimated_cost_usd": llm_response['estimated_cost_usd'], # Costo
            "llm_usage": llm_response['usage'],
            "model": llm_response['model'],
        }

# Ejemplo de uso (ejecutado en H9/app.py)
# rag_manager = RAGManager(index_dir='data/processed', embedding_model_name='sentence-transformers/all-MiniLM-L6-v2')
# response = rag_manager.synthesize(provider=chatgpt_provider, original_query="¿Cómo se apela una nota?", k=3)
