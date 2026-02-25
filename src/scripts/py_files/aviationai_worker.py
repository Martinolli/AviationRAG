import json
import logging
import sys
import traceback
import uuid

from aviationai import answer_query
from chat_db import retrieve_chat_from_db, store_chat_in_db


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def compact_sources(sources, max_sources=20, max_chars=3000):
    if not isinstance(sources, list):
        return []
    compacted = []
    for item in sources[:max_sources]:
        filename = str(item.get("filename", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not filename or not chunk_id:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        compacted.append(
            {
                "filename": filename,
                "chunk_id": chunk_id,
                "text": text,
            }
        )
    return compacted


def handle_ask(request_id, payload):
    message = str(payload.get("message", "")).strip()
    if not message:
        return {
            "id": request_id,
            "success": False,
            "error": "Field 'message' is required.",
        }

    session_id = str(payload.get("session_id", "")).strip() or str(uuid.uuid4())
    strict_mode = payload.get("strict_mode", None)
    target_document = str(payload.get("target_document", "")).strip() or None
    model = str(payload.get("model", "gpt-4-turbo")).strip() or "gpt-4-turbo"
    store = bool(payload.get("store", True))

    result = answer_query(
        query=message,
        model=model,
        strict_mode=strict_mode,
        target_filename=target_document,
    )

    if store:
        store_chat_in_db(session_id, message, result["answer"])

    return {
        "id": request_id,
        "success": True,
        "action": "ask",
        "session_id": session_id,
        "answer": result["answer"],
        "strict_mode": result["strict_mode"],
        "target_filename": result["target_filename"],
        "citations": result["citations"],
        "sources": compact_sources(result.get("sources", [])),
    }


def handle_history(request_id, payload):
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return {
            "id": request_id,
            "success": False,
            "error": "Field 'session_id' is required.",
        }

    limit = int(payload.get("limit", 10))
    limit = max(1, min(limit, 50))
    messages = retrieve_chat_from_db(session_id, limit=limit)

    return {
        "id": request_id,
        "success": True,
        "action": "history",
        "session_id": session_id,
        "messages": messages,
    }


def main():
    logging.basicConfig(level=logging.ERROR)
    emit({"event": "ready", "success": True})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None
        try:
            payload = json.loads(line)
            request_id = payload.get("id")
            if request_id is None:
                request_id = str(uuid.uuid4())

            action = payload.get("action")
            if action == "ask":
                response = handle_ask(request_id, payload)
            elif action == "history":
                response = handle_history(request_id, payload)
            elif action == "ping":
                response = {"id": request_id, "success": True, "action": "ping"}
            else:
                response = {
                    "id": request_id,
                    "success": False,
                    "error": f"Unsupported action: {action}",
                }
            emit(response)
        except Exception as error:
            emit(
                {
                    "id": request_id or str(uuid.uuid4()),
                    "success": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    main()
