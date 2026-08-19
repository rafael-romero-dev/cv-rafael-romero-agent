# openresponses_api — Endpoint Open Responses del Agente de CV

Endpoint HTTP compatible con la especificación abierta **Open Responses**
(`POST /v1/responses`) que expone el pipeline RAG del CV de Rafael Romero
Negrete. Ver la arquitectura general y las decisiones de diseño en el
[README raíz](../README.md).

## Requisitos previos

- Python 3.11+
- Una API key de Google AI Studio / Gemini con acceso a los modelos
  configurados (`gemini-3.5-flash`, `gemini-embedding-001`)
- La base vectorial ya generada en `../chroma_db/` (se crea corriendo
  `python vector_stores.py` desde la raíz del repo, una sola vez, con los
  PDFs de `../data/`)

## Instalación local

```bash
cd openresponses_api
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `GOOGLE_API_KEY` | Sí | — | API key de Gemini. Sin ella el servidor no puede generar embeddings ni respuestas. |
| `OPENRESPONSES_API_TOKEN` | No | (vacío) | Si se define, el endpoint exige `Authorization: Bearer <token>` en cada request. Si se deja vacío, el endpoint queda abierto (recomendado solo para pruebas). |
| `PUBLIC_MODEL_NAME` | No | `cv-rafael-romero-agent` | Nombre del modelo que se reporta en `/v1/models` y se espera en el campo `model` de las requests. |
| `CHROMA_DB_PATH` | No | `../chroma_db` (relativo) | Ruta a la base vectorial persistida. En Docker se fija a `/app/chroma_db`. |
| `EMBEDDING_MODEL`, `QUERY_MODEL`, `GENERATION_MODEL` | No | ver `config.py` | Modelos de Gemini usados para embeddings, generación de multi-query y generación de la respuesta final. |
| `ENABLE_HYBRID_SEARCH` | No | `true` | Activa el `EnsembleRetriever` (MMR+MultiQuery combinado con similarity search puro). |

Crea un archivo `.env` local (ya está en `.gitignore`, no se sube) o
exporta las variables en tu shell antes de correr el servidor.

## Correrlo en local

```bash
export GOOGLE_API_KEY="tu-api-key"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verifica que esté vivo:

```bash
curl http://127.0.0.1:8000/health
```

## Endpoints

- `GET /health` — healthcheck simple.
- `GET /v1/models` — lista el modelo publicado (respeta `OPENRESPONSES_API_TOKEN` si está configurado).
- `POST /v1/responses` — endpoint principal. Acepta `input` como string
  (turno único) o como lista de mensajes `{role, content}` (conversación
  con historial — ver `main.py::_extract_conversation`). No soporta
  `stream: true` todavía.