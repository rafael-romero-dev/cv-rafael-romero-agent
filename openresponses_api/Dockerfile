FROM python:3.11-slim

WORKDIR /app

# Dependencias del adaptador Open Responses
COPY openresponses_api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Código del adaptador + prompts reutilizados + base vectorial persistida
COPY openresponses_api ./openresponses_api
COPY asistente_CV_RAG/prompts.py ./asistente_CV_RAG/prompts.py
COPY chroma_db ./chroma_db

WORKDIR /app/openresponses_api

ENV CHROMA_DB_PATH=/app/chroma_db
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
