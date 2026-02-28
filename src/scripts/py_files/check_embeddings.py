"""Module for verifying and analyzing embeddings from a JSON file."""

import json
from collections import Counter, defaultdict

from config import EMBEDDINGS_FILE


def verify_embeddings(file_path):
    """Verify and analyze embeddings from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Error: Expected a list, but got {type(data)}")
            return

        print(f"Total number of embeddings: {len(data)}")

        if len(data) == 0:
            print("Error: The file is empty.")
            return

        # Check the structure of the first item
        first_item = data[0]
        expected_keys = {
            "chunk_id",
            "filename",
            "metadata",
            "text",
            "tokens",
            "embedding",
        }
        actual_keys = set(first_item.keys())

        if actual_keys != expected_keys:
            print(
                f"Warning: Keys mismatch. Expected {expected_keys}, got {actual_keys}"
            )

        # Analyze embeddings
        embedding_sizes = set()
        filenames = defaultdict(int)
        total_tokens = 0

        for item in data:
            embedding = item.get("embedding")
            if embedding:
                embedding_sizes.add(len(embedding))

            filenames[item.get("filename", "Unknown")] += 1
            total_tokens += item.get("tokens", 0)

        print(f"\nEmbedding sizes found: {embedding_sizes}")
        if len(embedding_sizes) > 1:
            print("Warning: Multiple embedding sizes detected.")

        print("\nFilenames and their embedding counts:")
        for filename, count in filenames.items():
            print(f"  {filename}: {count}")

        print(f"\nTotal tokens across all embeddings: {total_tokens}")
        print(f"Average tokens per embedding: {total_tokens / len(data):.2f}")

        # Check for duplicate chunk_ids
        chunk_ids = [
            item.get("chunk_id") for item in data if item.get("chunk_id") is not None
        ]
        chunk_id_counts = Counter(chunk_ids)
        duplicate_chunk_ids = {
            chunk_id for chunk_id, count in chunk_id_counts.items() if count > 1
        }
        if duplicate_chunk_ids:
            print(f"\nWarning: Found {len(duplicate_chunk_ids)} duplicate chunk_ids:")
            for chunk_id in duplicate_chunk_ids:
                print(f"  {chunk_id}")
        else:
            print("\nNo duplicate chunk_ids found.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
    except (TypeError, KeyError, ValueError) as e:
        print(f"An unexpected error occurred: {str(e)}")


# Use the path from your project structure
embeddings_path = EMBEDDINGS_FILE
verify_embeddings(embeddings_path)
