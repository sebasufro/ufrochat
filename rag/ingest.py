# rag/ingest.py

import os
import re
import argparse
from pypdf import PdfReader
import tiktoken
import pandas as pd
from tqdm import tqdm

# --- 2.1 Funciones de Lectura y Extracción ---

def read_pdf(path: str) -> str:
    """Extrae texto de un archivo PDF."""
    try:
        reader = PdfReader(path)
        # Extrae texto de cada página, filtrando las que son nulas
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return text
    except Exception as e:
        print(f"Error al leer PDF {path}: {e}")
        return ""

def read_txt(path: str) -> str:
    """Extrae texto de un archivo TXT."""
    try:
        # CORRECCIÓN CLAVE: Lee el contenido directamente con codificación UTF-8
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error al leer TXT {path}: {e}")
        return ""

def clean_and_normalize_text(text: str) -> str:
    """Limpia encabezados, pies de página y normaliza saltos de línea."""
    # 1. Eliminar múltiples saltos de línea (simulando limpieza de pies/encabezados por espacio)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # 2. Reemplazar espacios múltiples por un solo espacio (limpieza general)
    text = re.sub(r"\s{2,}", " ", text)
    # 3. Limpieza específica (si se detectan patrones comunes en normativa UFRO)
    # Esto es un placeholder; se ajustaría al analizar los PDFs reales.
    # text = re.sub(r"UNIVERSIDAD DE LA FRONTERA.*", "", text)
    return text.strip()

# --- 2.2 Función de Chunking por Tokens ---

def chunk_by_tokens(text: str, tokenizer: tiktoken.Encoding, 
                    chunk_size: int = 900, overlap: int = 120) -> list:
    """Divide un texto en chunks con solapamiento, basado en tokens."""
    ids = tokenizer.encode(text)
    chunks, start, cid = [], 0, 0
    
    while start < len(ids):
        # El fin del chunk es el mínimo entre el tamaño y el final del texto
        end = min(start + chunk_size, len(ids))
        sub_text = tokenizer.decode(ids[start:end])
        
        # Guardamos el chunk con sus metadatos de token
        chunks.append({
            "chunk_id": cid,
            "token_start": start,
            "token_end": end,
            "text": sub_text
        })
        
        # Si llegamos al final, salimos
        if end == len(ids):
            break
        
        # Movemos el inicio con solapamiento
        start = end - overlap
        cid += 1
        
    return chunks

# --- 2.3 Orquestación de Ingesta ---

def load_and_chunk_docs(data_dir: str, sources_csv: str, 
                        chunk_size: int, overlap: int) -> pd.DataFrame:
    """Carga documentos, los fragmenta y combina con metadatos de fuentes."""
    
    # 1. Cargar metadatos de fuentes
    # Esto asume que sources.csv contiene: doc_id, path, url, vigencia, title
    try:
        sources_df = pd.read_csv(sources_csv, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de fuentes en {sources_csv}")
        return pd.DataFrame()
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    all_chunks = []

    print(f"Procesando {len(sources_df)} documentos en {data_dir}...")
    
    for _, row in tqdm(sources_df.iterrows(), total=len(sources_df), desc="Chunking documentos"):
        doc_path = os.path.join(data_dir, row['path'])
        
        if not os.path.exists(doc_path):
            print(f"Archivo no encontrado en {doc_path}. Saltando.")
            continue
            
        # Determinar el lector según la extensión
        if doc_path.lower().endswith(".pdf"):
            raw_text = read_pdf(doc_path)
        # Podrías agregar 'elif doc_path.lower().endswith((".html", ".htm")):' aquí para HTML.
        else:
            print(f"Formato no soportado para {doc_path}. Saltando.")
            continue
            
        cleaned_text = clean_and_normalize_text(raw_text)
        
        if not cleaned_text:
            continue
            
        # Chunking
        chunks = chunk_by_tokens(cleaned_text, tokenizer, chunk_size, overlap)
        
        # Asignar metadatos del documento a cada chunk
        for chunk in chunks:
            chunk.update({
                "doc_id": row['doc_id'],
                "doc_title": row['title'],
                "doc_url": row['url'],
                "doc_vigencia": row['vigencia'],
                "source_path": doc_path,
            })
            all_chunks.append(chunk)

    return pd.DataFrame(all_chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta y Chunking de Documentos UFRO.")
    parser.add_argument("--data_dir", default="data/raw", help="Directorio con los archivos PDF/HTML.")
    parser.add_argument("--sources_csv", default="data/sources.csv", help="Ruta al archivo CSV con metadatos de las fuentes.")
    parser.add_argument("--out_parquet", default="data/processed/chunks.parquet", help="Ruta de salida para el archivo Parquet de chunks.")
    parser.add_argument("--chunk_size", type=int, default=900, help="Tamaño de chunk en tokens.")
    parser.add_argument("--overlap", type=int, default=120, help="Solapamiento de chunks en tokens.")
    args = parser.parse_args()
    
    df_chunks = load_and_chunk_docs(args.data_dir, args.sources_csv, args.chunk_size, args.overlap)

    if not df_chunks.empty:
        df_chunks.to_parquet(args.out_parquet, index=False)
        print(f"Ingesta finalizada. {len(df_chunks)} chunks guardados en {args.out_parquet}")
    else:
        print("No se generaron chunks. Verifica tus documentos y sources.csv.")
