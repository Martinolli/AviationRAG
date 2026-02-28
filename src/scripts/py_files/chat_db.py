"""Module for managing chat sessions and storing/retrieving chat history from AstraDB."""

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone

from config import CHAT_ID_DIR, JS_SCRIPTS_DIR

SESSION_METADATA_FILE = CHAT_ID_DIR / "session_metadata.json"
SESSION_INDEX_FILE = CHAT_ID_DIR / "session_index.json"


def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_json_object(output):
    """Extract the first JSON object from a string."""
    first_brace = output.find("{")
    if first_brace != -1:
        output = output[first_brace:]
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        logging.error("JSON parsing error: %s. Raw output: %s", error, output)
        return None


def _ensure_chat_id_dir():
    """Ensure the chat ID directory exists."""
    CHAT_ID_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path, fallback):
    """Load JSON data from a file, returning a fallback value on error."""
    try:
        if not path.exists():
            return fallback
        with open(path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        return loaded if isinstance(loaded, type(fallback)) else fallback
    except (FileNotFoundError, json.JSONDecodeError, IOError) as error:
        logging.error("Error reading %s: %s", path, error)
        return fallback


def _save_json(path, payload):
    """Save JSON data to a file."""
    _ensure_chat_id_dir()
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)


def _load_legacy_titles():
    """Load legacy session titles from the metadata file."""
    raw = _load_json(SESSION_METADATA_FILE, {})
    titles = {}
    for session_id, value in raw.items():
        if isinstance(value, str):
            title = value.strip()
        elif isinstance(value, dict):
            title = str(value.get("title", "")).strip()
        else:
            title = ""
        if title:
            titles[str(session_id)] = title
    return titles


def _save_legacy_titles(titles):
    """Save legacy session titles to the metadata file."""
    _save_json(SESSION_METADATA_FILE, titles)


def _load_session_index():
    """Load session index from the index file."""
    raw = _load_json(SESSION_INDEX_FILE, {})
    index = {}
    for session_id, value in raw.items():
        if not isinstance(value, dict):
            continue
        created_at = str(value.get("created_at", "")).strip()
        updated_at = str(value.get("updated_at", "")).strip()
        pinned = bool(value.get("pinned", False))
        index[str(session_id)] = {
            "created_at": created_at,
            "updated_at": updated_at,
            "pinned": pinned,
        }
    return index


def _save_session_index(index):
    """Save session index to the index file."""
    _save_json(SESSION_INDEX_FILE, index)


def _build_session_object(session_id, title, index_entry):
    """Build a session object combining title and index metadata."""
    created_at = index_entry.get("created_at", "")
    updated_at = index_entry.get("updated_at", "")
    if not created_at:
        created_at = updated_at or _now_iso()
    if not updated_at:
        updated_at = created_at
    return {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "pinned": bool(index_entry.get("pinned", False)),
    }


def upsert_session_metadata(session_id, title=None, pinned=None):
    """Upsert session metadata with optional title and pinned status."""
    session_id = str(session_id or "").strip()
    if not session_id:
        raise ValueError("session_id is required")

    titles = _load_legacy_titles()
    index = _load_session_index()
    now = _now_iso()

    cleaned_title = str(title or "").strip()
    if cleaned_title:
        titles[session_id] = cleaned_title
    elif session_id not in titles:
        titles[session_id] = "New Session"

    entry = index.get(session_id, {})
    created_at = str(entry.get("created_at", "")).strip() or now
    new_entry = {
        "created_at": created_at,
        "updated_at": now,
        "pinned": bool(entry.get("pinned", False)),
    }
    if pinned is not None:
        new_entry["pinned"] = bool(pinned)
    index[session_id] = new_entry

    _save_legacy_titles(titles)
    _save_session_index(index)
    return _build_session_object(session_id, titles[session_id], new_entry)


def list_sessions(search="", filter_mode="all", limit=50):
    """List sessions with optional search and filtering."""
    limit_value = max(1, min(int(limit), 200))
    search_value = str(search or "").strip().lower()
    mode = str(filter_mode or "all").strip().lower()

    titles = _load_legacy_titles()
    index = _load_session_index()

    all_ids = set(titles.keys()) | set(index.keys())
    sessions = []
    for session_id in all_ids:
        title = titles.get(session_id, f"Session {session_id[:8]}")
        entry = index.get(session_id, {})
        sessions.append(_build_session_object(session_id, title, entry))

    sessions = sorted(
        sessions,
        key=lambda item: (
            bool(item.get("pinned", False)),
            str(item.get("updated_at", "")),
        ),
        reverse=True,
    )

    if mode == "pinned":
        sessions = [item for item in sessions if item.get("pinned", False)]
    elif mode == "recent":
        sessions = [item for item in sessions if not item.get("pinned", False)]

    if search_value:
        sessions = [
            item
            for item in sessions
            if search_value in item.get("title", "").lower()
            or search_value in item.get("id", "").lower()
        ]

    return sessions[:limit_value]


def delete_session_metadata(session_id):
    """Delete session metadata for the given session_id."""
    session_id = str(session_id or "").strip()
    if not session_id:
        return False

    titles = _load_legacy_titles()
    index = _load_session_index()
    removed = False

    if session_id in titles:
        del titles[session_id]
        removed = True
    if session_id in index:
        del index[session_id]
        removed = True

    _save_legacy_titles(titles)
    _save_session_index(index)
    return removed


def store_chat_in_db(session_id, user_query, ai_response, print_success=False, log_success=False):
    """Store a chat exchange via the Node.js AstraDB bridge."""
    script_path = JS_SCRIPTS_DIR / "store_chat.js"

    if not ai_response or len(ai_response) < 10:
        logging.error("Invalid AI response detected. Storing default message.")
        ai_response = "AI response was incomplete or not available."

    chat_data = {
        "action": "store",
        "session_id": session_id,
        "user_query": user_query,
        "ai_response": ai_response,
    }

    try:
        result = subprocess.run(
            ["node", str(script_path), json.dumps(chat_data)],
            check=True,
            capture_output=True,
            text=True,
            cwd=JS_SCRIPTS_DIR,
        )
        upsert_session_metadata(session_id, title=user_query[:80])

        if print_success:
            print("Chat stored successfully in AstraDB!")
        if log_success:
            logging.info("Storing chat for session: %s | Query: %s...", session_id, user_query[:50])
        if result.stdout.strip():
            parsed = _extract_json_object(result.stdout.strip())
            if not parsed and "success" not in result.stdout:
                logging.debug(result.stdout.strip())
    except subprocess.CalledProcessError as error:
        logging.error("Error storing chat: %s", error)
        if error.stderr:
            logging.error("Store chat stderr: %s", error.stderr.strip())


def retrieve_chat_from_db(session_id, limit=5, warn_on_empty_session=False):
    """Retrieve session chat history via the Node.js AstraDB bridge."""
    script_path = JS_SCRIPTS_DIR / "store_chat.js"
    logging.info("Retrieving chat messages for session: %s", session_id)

    if not session_id.strip():
        if warn_on_empty_session:
            print("Warning: `session_id` is empty. Generating a new one...")
        session_id = str(uuid.uuid4())

    chat_data = {
        "action": "retrieve",
        "session_id": session_id,
        "limit": limit,
    }

    try:
        result = subprocess.run(
            ["node", str(script_path), json.dumps(chat_data)],
            capture_output=True,
            text=True,
            check=True,
            cwd=JS_SCRIPTS_DIR,
        )
        parsed = _extract_json_object(result.stdout.strip())
        if isinstance(parsed, dict) and parsed.get("success", False):
            return parsed.get("messages", [])
        logging.error("Chat retrieval failed. Parsed output: %s", parsed)
        return []
    except subprocess.CalledProcessError as error:
        logging.error("Error retrieving chat: %s", error)
        if error.stderr:
            logging.error("Retrieve chat stderr: %s", error.stderr.strip())
        return []


def delete_chat_session_in_db(session_id):
    """Delete all chat rows for one session in AstraDB."""
    script_path = JS_SCRIPTS_DIR / "store_chat.js"
    chat_data = {
        "action": "delete",
        "session_id": str(session_id).strip(),
    }

    try:
        result = subprocess.run(
            ["node", str(script_path), json.dumps(chat_data)],
            capture_output=True,
            text=True,
            check=True,
            cwd=JS_SCRIPTS_DIR,
        )
        parsed = _extract_json_object(result.stdout.strip())
        if isinstance(parsed, dict) and parsed.get("success", False):
            return int(parsed.get("deleted_rows", 0))
        return 0
    except subprocess.CalledProcessError as error:
        logging.error("Error deleting session in DB: %s", error)
        if error.stderr:
            logging.error("Delete session stderr: %s", error.stderr.strip())
        return 0
