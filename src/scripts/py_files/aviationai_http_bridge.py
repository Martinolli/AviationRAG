"""HTTP bridge server for Aviation RAG API communication.

This module provides an HTTP server that acts as a bridge between HTTP clients
and the Aviation AI worker, handling requests for chat, history, and session management.
"""
import hmac
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aviationai_worker import (
    handle_ask,
    handle_history,
    handle_session_delete,
    handle_session_upsert,
    handle_sessions_list,
)


def now_iso():
    """Return the current UTC time as an ISO 8601 formatted string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def bridge_token():
    """Retrieve and return the HTTP bridge authentication token from environment variables."""
    return str(os.getenv("AVIATION_API_HTTP_TOKEN", "")).strip()


def auth_ok(header_value):
    """Validate the authorization header against the configured token.
    
    Args:
        header_value: The Authorization header value to validate.
        
    Returns:
        bool: True if the header is valid or no token is configured, False otherwise.
    """
    token = bridge_token()
    if not token:
        return True
    if not header_value:
        return False
    prefix = "Bearer "
    if not header_value.startswith(prefix):
        return False
    candidate = header_value[len(prefix) :].strip()
    return hmac.compare_digest(candidate, token)


def dispatch(request_id, payload):
    action = str(payload.get("action", "")).strip()
    if action == "ask":
        return handle_ask(request_id, payload)
    if action == "history":
        return handle_history(request_id, payload)
    if action == "sessions_list":
        return handle_sessions_list(request_id, payload)
    if action == "session_upsert":
        return handle_session_upsert(request_id, payload)
    if action == "session_delete":
        return handle_session_delete(request_id, payload)
    if action == "ping":
        return {"id": request_id, "success": True, "action": "ping"}
    return {
        "id": request_id,
        "success": False,
        "error": f"Unsupported action: {action}",
    }


class Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the Aviation RAG API bridge.
    
    Handles GET and POST requests for health checks and command execution,
    including authentication and JSON payload processing.
    """
    server_version = "AviationBridge/1.0"

    def _write_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _is_authorized(self):
        auth_header = self.headers.get("Authorization")
        return auth_ok(auth_header)

    def do_GET(self):
        """Handle GET requests for health check endpoint."""
        if self.path != "/health":
            self._write_json(404, {"success": False, "error": "Not Found"})
            return
        self._write_json(
            200,
            {
                "success": True,
                "service": "aviation-api-http-bridge",
                "timestamp": now_iso(),
                "token_required": bool(bridge_token()),
            },
        )

    def do_POST(self):
        """Handle POST requests for command execution with authentication."""
        if self.path != "/command":
            self._write_json(404, {"success": False, "error": "Not Found"})
            return

        if not self._is_authorized():
            self._write_json(401, {"success": False, "error": "Unauthorized"})
            return

        try:
            payload = self._read_json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            self._write_json(400, {"success": False, "error": "Invalid JSON payload"})
            return

        if not isinstance(payload, dict):
            self._write_json(400, {"success": False, "error": "Payload must be a JSON object"})
            return

        request_id = str(payload.get("id") or uuid.uuid4())
        action = str(payload.get("action", "")).strip()
        if not action:
            self._write_json(
                400,
                {"id": request_id, "success": False, "error": "Field 'action' is required."},
            )
            return

        try:
            result = dispatch(request_id, payload)
            if (
                isinstance(result, dict)
                and result.get("success") is False
                and str(result.get("error", "")).startswith("Unsupported action:")
            ):
                self._write_json(400, result)
                return
            self._write_json(200, result)
        except (KeyError, ValueError, TypeError) as error:
            self._write_json(
                500,
                {
                    "id": request_id,
                    "success": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, format, *args):
        return


def main():
    """Start the HTTP bridge server for Aviation RAG API communication."""
    bind = str(os.getenv("AVIATION_API_HTTP_BIND", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(os.getenv("AVIATION_API_HTTP_PORT", "8010"))

    server = ThreadingHTTPServer((bind, port), Handler)
    print(
        json.dumps(
            {
                "event": "ready",
                "service": "aviation-api-http-bridge",
                "bind": bind,
                "port": port,
                "token_required": bool(bridge_token()),
            }
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
