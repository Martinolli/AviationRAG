import argparse
import json
import logging
import uuid

from aviationai import answer_query
from chat_db import retrieve_chat_from_db, store_chat_in_db


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def output(payload):
    print(json.dumps(payload, ensure_ascii=False))


def run_ask(args):
    session_id = args.session_id.strip() if args.session_id else str(uuid.uuid4())

    result = answer_query(
        query=args.message,
        model=args.model,
        strict_mode=args.strict_mode,
        target_filename=args.target_document,
    )

    if args.store:
        store_chat_in_db(session_id, args.message, result["answer"])

    output(
        {
            "success": True,
            "action": "ask",
            "session_id": session_id,
            "answer": result["answer"],
            "strict_mode": result["strict_mode"],
            "target_filename": result["target_filename"],
            "citations": result["citations"],
            "sources": result["sources"],
        }
    )


def run_history(args):
    session_id = args.session_id.strip()
    messages = retrieve_chat_from_db(session_id, limit=args.limit)
    output(
        {
            "success": True,
            "action": "history",
            "session_id": session_id,
            "messages": messages,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="AviationAI API bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Generate an answer for a single query")
    ask_parser.add_argument("--message", required=True, help="User query")
    ask_parser.add_argument("--session-id", default="", help="Session ID")
    ask_parser.add_argument("--model", default="gpt-4-turbo", help="OpenAI model name")
    ask_parser.add_argument(
        "--strict-mode",
        type=str_to_bool,
        default=None,
        nargs="?",
        const=True,
        help="Force strict mode true/false (optional)",
    )
    ask_parser.add_argument("--target-document", default="", help="Target filename for grounded mode")
    ask_parser.add_argument(
        "--store",
        type=str_to_bool,
        default=True,
        help="Store chat in Astra conversation history",
    )

    history_parser = subparsers.add_parser("history", help="Retrieve chat history")
    history_parser.add_argument("--session-id", required=True, help="Session ID")
    history_parser.add_argument("--limit", type=int, default=10, help="Maximum history entries")

    args = parser.parse_args()

    try:
        if args.command == "ask":
            run_ask(args)
            return
        if args.command == "history":
            run_history(args)
            return
        output({"success": False, "error": f"Unsupported command: {args.command}"})
    except Exception as error:
        logging.exception("aviationai_api error")
        output({"success": False, "error": str(error)})


if __name__ == "__main__":
    main()
