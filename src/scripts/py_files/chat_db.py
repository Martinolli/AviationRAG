import json
import logging
import subprocess
import uuid

from config import JS_SCRIPTS_DIR


def _parse_json_output(output):
    first_brace = output.find("{")
    if first_brace != -1:
        output = output[first_brace:]

    try:
        parsed_output = json.loads(output)
        if parsed_output.get("success", False):
            return parsed_output.get("messages", [])
        logging.error(f"Chat retrieval failed. Parsed output: {parsed_output}")
        return []
    except json.JSONDecodeError as error:
        logging.error(f"JSON parsing error: {error}. Raw output: {output}")
        return []


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

        if print_success:
            print("Chat stored successfully in AstraDB!")
        if log_success:
            logging.info(f"Storing chat for session: {session_id} | Query: {user_query[:50]}...")
        if result.stdout.strip():
            parsed = _parse_json_output(result.stdout.strip())
            if parsed == [] and "success" not in result.stdout:
                logging.debug(result.stdout.strip())
    except subprocess.CalledProcessError as error:
        logging.error(f"Error storing chat: {error}")
        if error.stderr:
            logging.error(f"Store chat stderr: {error.stderr.strip()}")


def retrieve_chat_from_db(session_id, limit=5, warn_on_empty_session=False):
    """Retrieve session chat history via the Node.js AstraDB bridge."""
    script_path = JS_SCRIPTS_DIR / "store_chat.js"
    logging.info(f"Retrieving chat messages for session: {session_id}")

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
        return _parse_json_output(result.stdout.strip())
    except subprocess.CalledProcessError as error:
        logging.error(f"Error retrieving chat: {error}")
        if error.stderr:
            logging.error(f"Retrieve chat stderr: {error.stderr.strip()}")
        return []
