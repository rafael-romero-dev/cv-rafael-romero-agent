"""
Adaptador Open Responses para el agente de CV de Rafael Romero Negrete.

Expone:
  POST /v1/responses   -> endpoint principal, spec Open Responses (openresponses.org)
  GET  /v1/models       -> lista el modelo publicado, por compatibilidad
  GET  /health          -> healthcheck simple para el proveedor de hosting
"""

import time
import uuid
from typing import Any, List, Optional, Union

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from rag_core import answer_question

app = FastAPI(title="CV Agent - Open Responses Adapter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Esquemas (subconjunto del spec Open Responses que usamos) ----------

class InputMessage(BaseModel):
    type: Optional[str] = "message"
    role: str
    content: Union[str, List[Any]]


class ResponsesRequest(BaseModel):
    model: str
    input: Union[str, List[InputMessage]]
    stream: Optional[bool] = False
    previous_response_id: Optional[str] = None
    store: Optional[bool] = False


# ---------- Utilidades ----------

def _check_auth(authorization: Optional[str]):
    if not config.API_BEARER_TOKEN:
        return  # sin token configurado -> endpoint abierto (uso en pruebas/demo)
    expected = f"Bearer {config.API_BEARER_TOKEN}"
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid or missing bearer token",
                               "type": "invalid_request", "code": "unauthorized"}},
        )


def _extract_conversation(payload: ResponsesRequest) -> tuple[str, str]:
    """Separa el `input` del request en (historial_formateado, pregunta_actual).

    El spec Open Responses permite que `input` sea un string simple (sin
    historial) o una lista de mensajes con roles user/assistant (turnos
    previos de la conversación). Aquí se usa el último mensaje de rol
    'user' como la pregunta a responder, y todo lo anterior como contexto
    de historial para que el agente pueda resolver referencias como
    "cuéntame más" sin mantener estado en el servidor.
    """
    if isinstance(payload.input, str):
        return "", payload.input

    def _text_of(content) -> str:
        if isinstance(content, str):
            return content
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("input_text", "text"):
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    # Encuentra el índice del último mensaje de usuario -> es la pregunta actual
    last_user_idx = None
    for idx in range(len(payload.input) - 1, -1, -1):
        if payload.input[idx].role == "user":
            last_user_idx = idx
            break

    if last_user_idx is None:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "No user message found in 'input'",
                               "type": "invalid_request", "param": "input",
                               "code": "invalid_input"}},
        )

    question = _text_of(payload.input[last_user_idx].content)

    history_lines = []
    for item in payload.input[:last_user_idx]:
        role_label = "Usuario" if item.role == "user" else "Agente"
        text = _text_of(item.content)
        if text:
            history_lines.append(f"{role_label}: {text}")

    history = "\n".join(history_lines)
    return history, question


def _build_response_object(response_id: str, model: str, text: str) -> dict:
    now = int(time.time())
    msg_id = f"msg_{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "previous_response_id": None,
        "usage": None,
    }


# ---------- Rutas ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": config.PUBLIC_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rafael-romero-negrete",
            }
        ],
    }


@app.post("/v1/responses")
async def create_response(payload: ResponsesRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)

    if payload.stream:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Streaming (stream=true) is not supported by this agent yet.",
                               "type": "invalid_request", "param": "stream",
                               "code": "unsupported_parameter"}},
        )

    history, question = _extract_conversation(payload)

    try:
        answer_text = answer_question(question, history=history)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Error procesando la consulta: {exc}",
                               "type": "model_error", "code": "generation_failed"}},
        )

    response_id = f"resp_{uuid.uuid4().hex}"
    return _build_response_object(response_id, payload.model, answer_text)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error", "code": "internal_error"}},
    )
