# app.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from loguru import logger
from typing import Dict, Any, List

# Agrega la raíz del proyecto para importar módulos (providers, rag)
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# --- Local Modules Imports ---
from providers.base import LLMProvider
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import RAGManager # RAGManager ahora incluye el método .synthesize completo
# NO necesitamos importar format_context ni get_system_prompt, ya que .synthesize() los usa internamente.

# Cargar variables de entorno (para claves API)
load_dotenv()

# --- Configuración Constantes ---
INDEX_DIR = "data/processed" 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
DEFAULT_LLM = "chatgpt"

# --- Flask App Initialization ---
app = Flask(__name__)
# Configuración básica de Loguru
logger.add(sys.stderr, level="INFO", colorize=True, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

# --- Proveedores y Retriever Initialization ---
PROVIDER_MAP: Dict[str, Any] = {
    "chatgpt": ChatGPTProvider,
    "deepseek": DeepSeekProvider,
}
# Carga diferida (Lazy Loading)
retriever: RAGManager = None # type: ignore

def get_retriever() -> RAGManager:
    """Inicializa y retorna el RAGManager, cargándolo solo una vez con los parámetros requeridos."""
    global retriever
    if retriever is None:
        logger.info(f"Initializing RAGManager for the first time with index_dir='{INDEX_DIR}' and model='{EMBEDDING_MODEL}'...")
        try:
            # CORRECCIÓN CRUCIAL: Se pasan los directorios y modelo requeridos
            retriever = RAGManager(index_dir=INDEX_DIR, embedding_model_name=EMBEDDING_MODEL)
            logger.info("RAGManager initialized successfully.")
        except Exception as e:
            logger.error(f"ERROR CRÍTICO: No se pudo inicializar RAGManager. Verifique que los archivos de índice existan en {INDEX_DIR}. Error: {e}")
            # Lanzamos un error de ejecución para que el Flask lo maneje.
            raise RuntimeError("Fallo al cargar el índice FAISS.") from e
    return retriever

# --- Funciones Auxiliares para HTML ---

def format_citations_for_html(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Extrae fuentes únicas del resultado RAG y las formatea como una lista HTML."""
    if not retrieved_chunks:
        return ""
        
    # Usar un set para almacenar tuplas (title, url) y asegurar unicidad de fuentes
    unique_sources = set((chunk['doc_title'], chunk['doc_url']) for chunk in retrieved_chunks)
    
    html_list = "<h4>Fuentes Citadas (Documentos de Origen):</h4><ul>"
    for title, url in unique_sources:
        # Se asume que el URL es una URL de vigencia pública (H1)
        html_list += f'<li><a href="{url}" target="_blank">{title}</a></li>'
    html_list += "</ul>"
    return html_list

# --- Plantilla HTML (Mejorada para mostrar métricas y citas) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asistente Normativa UFRO (RAG)</title>
    <style>
        /* Estilos CSS adaptados */
        body { font-family: sans-serif; margin: 2em; background-color: #f4f7f6; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        form { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }
        input[type="text"], select { width: 100%; padding: 10px; margin: 5px 0 15px 0; display: inline-block; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        input[type="submit"] { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; float: right; }
        input[type="submit"]:hover { background-color: #0056b3; }
        .answer-box { background-color: #e9f7ff; padding: 15px; border-radius: 4px; border-left: 5px solid #007bff; clear: both; margin-top: 20px; }
        .meta { font-size: 0.85em; color: #666; margin-top: 10px; border-top: 1px dashed #ddd; padding-top: 10px; }
        .meta h4 { margin-top: 0; color: #0056b3; }
        .error { color: red; font-weight: bold; background-color: #ffe9e9; border-left: 5px solid red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Asistente de Normativa UFRO</h1>
        <p>Consulta reglamentos, el calendario académico y normativas de convivencia.</p>
        
        <form method="POST">
            <label for="question">Pregunta:</label>
            <input type="text" id="question" name="question" value="{{ question or '' }}" placeholder="Ej: ¿Cuál es el plazo para la apelación de notas?" required>

            <label for="provider">Proveedor LLM:</label>
            <select id="provider" name="provider">
                <option value="chatgpt" {{ 'selected' if provider == 'chatgpt' else '' }}>ChatGPT (GPT-3.5-Turbo)</option>
                <option value="deepseek" {{ 'selected' if provider == 'deepseek' else '' }}>DeepSeek-Chat</option>
            </select>

            <input type="submit" value="Consultar">
        </form>

        {% if answer %}
            <div class="answer-box">
                <h2>Respuesta del Chatbot ({{ provider | upper }}):</h2>
                <p>{{ answer | safe }}</p>
                
                <div class="meta">
                    {% if citations %}
                        {{ citations | safe }}
                    {% endif %}
                    <p>
                        <strong>Trazabilidad:</strong> 
                        Latencia Total: <strong>{{ latency }}s</strong> | 
                        Costo Estimado: <strong>USD {{ cost }}</strong>
                    </p>
                </div>
            </div>
        {% endif %}
        
        {% if error_message %}
            <div class="answer-box error">
                <h2>Error en la Consulta:</h2>
                <p>{{ error_message }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Maneja la lógica de la consulta RAG y renderiza la plantilla HTML."""
    
    question = request.form.get('question', '')
    provider_name = request.form.get('provider', DEFAULT_LLM).lower()
    
    # Inicialización de variables para el template
    answer = None
    citations = None
    latency = "N/A"
    cost = "N/A"
    error_message = None

    if request.method == 'POST' and question:
        try:
            current_retriever = get_retriever()
            
            provider_class = PROVIDER_MAP.get(provider_name)
            if not provider_class:
                error_message = f"Error: Proveedor '{provider_name}' no es válido."
            else:
                provider: LLMProvider = provider_class()
                
                # Verificación de clave API (H7)
                api_key_env = "OPENAI_API_KEY" if provider_name == "chatgpt" else "DEEPSEEK_API_KEY"
                if not os.getenv(api_key_env):
                    error_message = f"Error: La variable de entorno **{api_key_env}** no está configurada."
                    logger.error(error_message)
                else:
                    logger.info(f"Received question: '{question}' for provider: {provider.name}")
                    
                    # PASO CRUCIAL: Uso del método sintetizado (RAG Pipeline H5/H7/H8)
                    rag_response = current_retriever.synthesize(
                        provider=provider,
                        original_query=question,
                        k=5, 
                        use_rewrite=False # Desactivamos rewrite para menor latencia web
                    )
                    
                    # Extracción de resultados y métricas
                    answer = rag_response['answer'].replace('\n', '<br>') # Formatear saltos de línea para HTML
                    latency = f"{rag_response['total_latency_sec']:.2f}"
                    cost = f"{rag_response['estimated_cost_usd']:.5f}"
                    citations = format_citations_for_html(rag_response['retrieved_chunks'])

        except RuntimeError as e:
            # Captura el error específico del get_retriever (problema de FAISS/index)
            error_message = f"Error crítico al iniciar el sistema RAG: **{str(e)}**. Verifique el índice FAISS y el directorio {INDEX_DIR}."
            logger.error(error_message)
        except Exception as e:
            error_message = f"Error al contactar al proveedor **{provider_name}**: {e}"
            logger.error(f"General processing error: {e}")
            
    # Renderizar la plantilla
    return render_template_string(
        HTML_TEMPLATE, 
        question=question, 
        provider=provider_name, 
        answer=answer, 
        citations=citations, 
        latency=latency,
        cost=cost,
        error_message=error_message
    )


# --- Initialization para desarrollo local / AWS EC2 ---

if __name__ == "__main__":
    # Si estás en EC2, recuerda usar 'gunicorn -w 4 'app:app' --bind 0.0.0.0:8080'
    logger.info("Starting Flask server for local development on http://0.0.0.0:8080")
    
    # Intenta cargar el retriever una vez al inicio para verificar fallos fatales
    try:
        get_retriever()
    except Exception:
        # El error es manejado en la ruta de Flask.
        pass
        
    app.run(host="0.0.0.0", port=8080, debug=True)