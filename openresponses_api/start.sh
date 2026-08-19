#!/bin/sh
set -e

if [ ! -d "/app/chroma_db" ]; then
    echo "Creando base vectorial Chroma..."
    python /app/vector_stores.py
else
    echo "Base vectorial Chroma encontrada."
fi

echo "Iniciando FastAPI..."
cd /app/openresponses_api
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"