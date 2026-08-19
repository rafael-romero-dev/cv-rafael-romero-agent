# Configuración de modelos
EMBEDDING_MODEL = "gemini-embedding-001"
QUERY_MODEL = "gemini-3.5-flash"
GENERATION_MODEL = "gemini-3.5-flash"

# Configuración del vector store
CHROMA_DB_PATH = "C:\\Users\\rafae\\curso_langchain\\CV_Agente\\chroma_db"

# Configuración del retriever
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70