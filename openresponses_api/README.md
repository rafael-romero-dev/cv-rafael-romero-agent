# Agente de CV — Rafael Romero Negrete

Agente conversacional que responde preguntas sobre la trayectoria profesional,
experiencia, formación, habilidades y proyectos de Rafael Romero Negrete

## Arquitectura

```
CV_Agente/
├── data/                     # PDFs fuente: CV, certificaciones, proyectos, constancias
├── vector_stores.py          # Script de ingesta: carga PDFs, chunking, embeddings -> Chroma
├── chroma_db/                # Base vectorial persistida (Chroma)
├── asistente_CV_RAG/         # Interfaz de chat en Streamlit para uso/depuración local
│   ├── app.py
│   ├── rag_system.py
│   ├── config.py
│   └── prompts.py
└── openresponses_api/        # Endpoint HTTP compatible con Open Responses
    ├── main.py                # Servidor FastAPI, POST /v1/responses
    ├── rag_core.py            # Mismo pipeline RAG, desacoplado de Streamlit
    ├── config.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md               # Detalle de decisiones técnicas del endpoint
```

## Cómo funciona el agente (resumen técnico)

1. **Ingesta** (`vector_stores.py`): los documentos del CV (PDF) se cargan,
   se dividen en chunks (`chunk_size=3000`, `overlap=500`) y se embeben con
   `gemini-embedding-001` en una base vectorial Chroma persistida en disco.
2. **Recuperación**: sobre esa base se combina un retriever **MMR** (Maximal
   Marginal Relevance, para diversidad de resultados) con un
   **MultiQueryRetriever** (que genera 3 variantes de la pregunta del usuario
   para mejorar el recall) y, opcionalmente, un **EnsembleRetriever** híbrido
   que combina MMR con búsqueda por similaridad pura.
3. **Generación**: los fragmentos recuperados se inyectan en un prompt
   (`prompts.py`) que instruye al modelo (`gemini-3.5-flash`) a responder
   *solo* con base en los documentos, citar la fuente, y evitar inventar
   información — clave para que el agente sea confiable en un proceso de
   selección.
4. **Dos interfaces sobre el mismo motor**:
   - `asistente_CV_RAG/` — una UI de chat en Streamlit para pruebas
     manuales y demo visual.
   - `openresponses_api/` — un endpoint HTTP (`POST /v1/responses`) que
     implementa la especificación abierta **Open Responses**

## Por qué esta arquitectura (criterio de diseño)

- **RAG en vez de fine-tuning o prompt gigante con todo el CV**: los
  documentos fuente (CV, certificados, artículo de congreso, proyectos)
  cambian con el tiempo y son heterogéneos en formato; un enfoque RAG
  permite actualizar el conocimiento del agente simplemente re-ingiriendo
  nuevos PDFs, sin reentrenar ni reescribir prompts.
- **MMR + MultiQuery en vez de similarity search simple**: las preguntas de
  reclutadores sobre un CV suelen formularse de formas muy distintas
  ("¿qué experiencia tiene en IA?" vs "cuéntame de sus proyectos de Desarrollo 
  de Software"); MultiQuery mitiga ese desajuste léxico, y MMR evita
  que las respuestas se apoyen en fragmentos redundantes del mismo
  documento.
- **Separación entre motor RAG y las dos interfaces que lo consumen**: el
  mismo pipeline de recuperación y el mismo prompt alimentan tanto la app
  de Streamlit como el endpoint Open Responses, para que ambas superficies
  respondan de forma consistente. 
- **Verificación de coherencia**: antes de desplegar, se probó el endpoint
  localmente con `TestClient` (mockeando el LLM) para validar el contrato
  HTTP, y después con Gemini real para confirmar que las respuestas citan
  correctamente las fuentes y no alucinan información fuera de los
  documentos.

## Cómo correrlo

Ver instrucciones detalladas de instalación, variables de entorno y
despliegue en [`openresponses_api/README.md`](openresponses_api/README.md).

## Stack

Python · LangChain · ChromaDB · Google Gemini (`gemini-3.5-flash`,
`gemini-embedding-001`) · FastAPI · Streamlit · Docker
