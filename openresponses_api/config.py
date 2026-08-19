import os

# --- Modelos (Google Gemini) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
QUERY_MODEL = os.getenv("QUERY_MODEL", "gemini-3.5-flash")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash")

# Nombre público que el cliente Open Responses usará en el campo "model"
PUBLIC_MODEL_NAME = os.getenv("PUBLIC_MODEL_NAME", "cv-rafael-romero-agent")

# --- Vector store ---
# Ruta relativa al proyecto (portátil entre Windows/Linux/contenedor),
# en vez de la ruta absoluta de Windows que traía el config.py original.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(_THIS_DIR), "chroma_db"),
)

# --- Retriever ---
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "mmr")
MMR_DIVERSITY_LAMBDA = float(os.getenv("MMR_DIVERSITY_LAMBDA", "0.7"))
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "20"))
SEARCH_K = int(os.getenv("SEARCH_K", "2"))

ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))

# --- Auth del endpoint (Open Responses exige Authorization: Bearer <token>) ---
# Si se deja vacío, el servidor NO exige autenticación (útil para pruebas locales).
API_BEARER_TOKEN = os.getenv("OPENRESPONSES_API_TOKEN", "")

# --- Google API key para Gemini ---
# GOOGLE_API_KEY esta como variable de entorno (no se hardcodea aquí).
