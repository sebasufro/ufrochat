# rag/embed.py

import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# --- 3.1 Función de Embeddings ---

def embed_texts(model: SentenceTransformer, texts: list) -> np.ndarray:
    """Genera embeddings normalizados para una lista de textos."""
    # Usamos la misma configuración que en el tutorial (convert_to_numpy, normalize_embeddings)
    return model.encode(
        texts, 
        convert_to_numpy=True, 
        normalize_embeddings=True, 
        show_progress_bar=True
    ).astype("float32")

# --- 3.2 Función de Creación y Persistencia de Índice FAISS ---

def create_and_save_faiss_index(df: pd.DataFrame, out_dir: str, model_name: str):
    """Crea embeddings, construye un índice FAISS y guarda los archivos."""
    
    print(f"Cargando modelo de embeddings: {model_name}...")
    model = SentenceTransformer(model_name)
    
    texts = df['text'].tolist()
    
    print(f"Generando embeddings para {len(texts)} chunks...")
    embeddings = embed_texts(model, texts)
    
    # 1. Construir Índice FAISS
    dim = embeddings.shape[1]
    # Se recomienda IndexFlatIP (Inner Product) para embeddings normalizados (similitud coseno)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # 2. Guardar archivos
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.faiss")
    faiss_metadata_path = os.path.join(out_dir, "chunks.parquet") # El DF de chunks se guarda aquí

    # Guardar el índice FAISS
    faiss.write_index(index, index_path)
    print(f"Índice FAISS guardado en {index_path}")
    
    # Guardar los metadatos/chunks. La ruta es la misma que la de entrada, pero se re-guarda para asegurar consistencia
    df.to_parquet(faiss_metadata_path, index=False)
    print(f"Metadatos de chunks guardados en {faiss_metadata_path}")
    
    print(f"Dimensión del Embedding: {dim}")
    print(f"Total de Vectores en el Índice: {index.ntotal}")

# --- 3.3 Función Principal ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creación de Embeddings y Índice FAISS.")
    parser.add_argument("--in_parquet", default="data/processed/chunks.parquet", help="Ruta del archivo Parquet de chunks.")
    parser.add_argument("--out_dir", default="data/processed", help="Directorio de salida para index.faiss y chunks.parquet.")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modelo de sentence-transformers a usar.")
    args = parser.parse_args()
    
    try:
        df_chunks = pd.read_parquet(args.in_parquet)
        if df_chunks.empty:
            print("El archivo Parquet está vacío. Ejecuta primero rag/ingest.py.")
        else:
            create_and_save_faiss_index(df_chunks, args.out_dir, args.model)
            
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de chunks en {args.in_parquet}. Ejecuta primero rag/ingest.py.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
