"""
Núcleo del sistema RAG del CV, desacoplado de Streamlit.

Reutiliza la misma arquitectura de recuperación (MMR + MultiQueryRetriever +
Ensemble híbrido) del directorio asistente_CV_RAG/rag_system.py, pero:
  - No depende de `st.cache_resource` (se usa un singleton simple en memoria,
    válido para un servidor FastAPI de un solo proceso).
"""

import os
import sys
import threading

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever

import config

# Reutilizamos los prompts ya afinados en asistente_CV_RAG/prompts.py
_ASISTENTE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asistente_CV_RAG")
if _ASISTENTE_DIR not in sys.path:
    sys.path.insert(0, _ASISTENTE_DIR)
from prompts import RAG_TEMPLATE, MULTI_QUERY_PROMPT

_lock = threading.Lock()
_rag_chain = None
_retriever = None


def _build_rag_system():
    vectorstore = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL),
        persist_directory=config.CHROMA_DB_PATH,
    )

    llm_queries = ChatGoogleGenerativeAI(model=config.QUERY_MODEL, temperature=0)
    llm_generation = ChatGoogleGenerativeAI(model=config.GENERATION_MODEL, temperature=0)

    base_retriever = vectorstore.as_retriever(
        search_type=config.SEARCH_TYPE,
        search_kwargs={
            "k": config.SEARCH_K,
            "lambda_mult": config.MMR_DIVERSITY_LAMBDA,
            "fetch_k": config.MMR_FETCH_K,
        },
    )

    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.SEARCH_K},
    )

    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt,
    )

    if config.ENABLE_HYBRID_SEARCH:
        final_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3],
        )
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            if doc.metadata:
                if "source" in doc.metadata:
                    source = doc.metadata["source"]
                    source = source.split("\\")[-1] if "\\" in source else source
                    header += f" - Fuente: {source}"
                if "page" in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"
            formatted.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": (lambda x: x["question"]) | final_retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", "(sin mensajes previos)"),
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever


def get_rag_chain():
    """Singleton thread-safe: inicializa el pipeline solo una vez por proceso."""
    global _rag_chain, _retriever
    if _rag_chain is None:
        with _lock:
            if _rag_chain is None:
                _rag_chain, _retriever = _build_rag_system()
    return _rag_chain, _retriever


def answer_question(question: str, history: str = "") -> str:
    """Punto de entrada usado por el endpoint Open Responses.

    `history` es el historial de la conversación ya formateado como texto
    (turnos previos de usuario/asistente), usado por el prompt para resolver
    referencias como "cuéntame más" sin necesidad de mantener estado en el
    servidor: el cliente Open Responses reenvía los turnos anteriores en
    cada request (vía el campo `input`), y este servidor los formatea aquí.
    """
    rag_chain, _ = get_rag_chain()
    return rag_chain.invoke({"question": question, "history": history or "(sin mensajes previos)"})
