FROM python:3.11-slim

WORKDIR /app

COPY openresponses_api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY openresponses_api ./openresponses_api

COPY asistente_CV_RAG/prompts.py ./asistente_CV_RAG/prompts.py

COPY data ./data

COPY vector_stores.py ./vector_stores.py

ENV CHROMA_DB_PATH=/app/chroma_db

RUN chmod +x /app/openresponses_api/start.sh

WORKDIR /app/openresponses_api

EXPOSE 8000

CMD ["/app/openresponses_api/start.sh"]