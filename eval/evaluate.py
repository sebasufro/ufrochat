# eval/evaluate.py

import argparse
import json
import time
import pandas as pd
from typing import List, Dict, Any

# Librerías necesarias para métricas
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Clases y módulos creados
from providers.base import LLMProvider
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import RAGManager

# --- 6.2.1 Métricas de Calidad ---

def calculate_semantic_similarity(model: SentenceTransformer, expected: str, actual: str) -> float:
    """Calcula la similitud coseno entre la respuesta esperada y la real (Semántica)."""
    sentences = [expected, actual]
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    # Similitud coseno entre los dos vectores
    return cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]

def check_citation_coverage(response_text: str, expected_refs: List[Dict[str, str]]) -> float:
    """Verifica si las citas esperadas están presentes en el texto (Cobertura)."""
    if not expected_refs: # Si no se espera referencia, la cobertura es 1.0 si no se cita
        return 1.0 if "No encontrado en normativa UFRO" in response_text else 0.0

    expected_titles = [ref['title'] for ref in expected_refs]

    # Simple check: buscar títulos esperados en el texto
    present_citations = 0
    for title in expected_titles:
        if title in response_text:
            present_citations += 1

    return present_citations / len(expected_titles)

def calculate_precision_at_k(retrieved_chunks: List[Dict[str, Any]], expected_refs: List[Dict[str, str]]) -> float:
    """Calcula Precisión@k del recuperador (Relevancia de los chunks)."""
    if not expected_refs: return 1.0 # Si es OOD, cualquier chunk no es 'incorrecto' en este modelo simple.

    expected_urls = [ref['url'] for ref in expected_refs]
    relevant_retrieved = 0

    # Un chunk es relevante si su URL de origen coincide con una URL esperada
    for chunk in retrieved_chunks:
        if chunk['doc_url'] in expected_urls:
            relevant_retrieved += 1

    # Precisión: Proporción de chunks recuperados que son relevantes
    return relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0.0

# --- 6.2.2 Función de Evaluación Principal ---

def evaluate_gold_set(rag_manager: RAGManager, gold_set_path: str, providers: List[LLMProvider], k: int):
    """Ejecuta la evaluación del Gold Set en modo batch."""

    # Modelo para métricas semánticas (puede ser el mismo que el RAG)
    metric_model = SentenceTransformer(rag_manager.embed_model.get_sentence_feature_extractor().get_config()['model_name'])

    with open(gold_set_path, 'r', encoding='utf-8') as f:
        gold_set = [json.loads(line) for line in f]

    results = []

    for item in tqdm(gold_set, desc="Evaluando Gold Set"):
        query = item['query']
        expected_answer = item['expected_answer']
        expected_refs = item['references']

        for provider in providers:
            start_time = time.time()

            # 1. Ejecutar RAG Pipeline
            try:
                rag_response = rag_manager.synthesize(provider, query, k=k, use_rewrite=False)
            except Exception as e:
                print(f"Error en RAG para {provider.name} | QID {item['id']}: {e}")
                continue # Saltar si la API falla

            latency = time.time() - start_time
            actual_answer = rag_response['answer']

            # 2. Calcular Métricas
            sem_sim = calculate_semantic_similarity(metric_model, expected_answer, actual_answer)
            cite_coverage = check_citation_coverage(actual_answer, expected_refs)
            prec_at_k = calculate_precision_at_k(rag_response['retrieved_chunks'], expected_refs)

            # 3. Almacenar Resultado
            result = {
                "qid": item['id'],
                "query": query,
                "provider": provider.name,
                "model": rag_response['model'],
                "latency_sec": latency,
                "sem_similarity": sem_sim,
                "citation_coverage": cite_coverage,
                "precision_at_k": prec_at_k,
                "total_latency_sec": rag_response['total_latency_sec'],
                "retrieval_latency_sec": rag_response['retrieval_latency_sec'],
                "llm_latency_sec": rag_response['llm_latency_sec'],
                "estimated_cost_usd": rag_response['estimated_cost_usd'],
                "total_tokens": rag_response['llm_usage'].get('total_tokens', 0),
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "retrieval_chunks_count": len(rag_response['retrieved_chunks']),
                "total_tokens": rag_response['llm_usage'].get('total_tokens', 0)
            }
            results.append(result)

    # Guardar resultados en CSV
    df_results = pd.DataFrame(results)
    out_file = "eval/evaluation_results.csv"
    df_results.to_csv(out_file, index=False)
    print(f"\n Evaluación finalizada. Resultados guardados en {out_file}")

    # Imprimir resumen de métricas
    summary = df_results.groupby('provider')[['sem_similarity', 'citation_coverage', 'precision_at_k', 'latency_sec']].mean()
    print("\n--- Resumen de Métricas Promedio ---")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Evaluación Batch para Chatbot UFRO RAG.")
    parser.add_argument("--gold_set_path", default="eval/gold_set.jsonl", help="Ruta al archivo gold set.")
    parser.add_argument("--index_dir", default="data/processed", help="Directorio del índice FAISS.")
    parser.add_argument("--k", type=int, default=5, help="Top-K chunks a recuperar.")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modelo de embeddings.")
    args = parser.parse_args()

    # Inicialización
    rag_manager = RAGManager(args.index_dir, args.embed_model)
    providers = [
        ChatGPTProvider(model="gpt-3.5-turbo"),
        DeepSeekProvider(model="deepseek-chat")
    ]

    evaluate_gold_set(rag_manager, args.gold_set_path, providers, args.k)
