import os

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Rutas relativas al proyecto (portátiles entre Windows/Linux/contenedor)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(_BASE_DIR, "chroma_db")

loader = PyPDFDirectoryLoader(DATA_DIR)
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el directorio.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=500
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")

vectorstore = Chroma.from_documents(
    docs_split,
    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory=CHROMA_DB_PATH
)

consulta = "¿Qué proyectos en java tiene Rafael Romero Negrete?"

resultados = vectorstore.similarity_search(consulta, k=2)

print("Top 2 documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")