import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_FILES_DIR = ROOT / "src" / "scripts" / "py_files"
sys.path.insert(0, str(PY_FILES_DIR))

from extract_pkl_to_json import extract_pkl_to_json  # noqa: E402


class TestExtractPklToJson(unittest.TestCase):
    def test_extract_writes_expected_json(self):
        sample_corpus = [
            {
                "filename": "sample.docx",
                "text": "hello",
                "tokens": ["hello"],
                "metadata": {"source": "unit-test"},
                "category": "test",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pkl_path = tmp_path / "sample.pkl"
            json_path = tmp_path / "nested" / "sample.json"

            with open(pkl_path, "wb") as file:
                pickle.dump(sample_corpus, file)

            extract_pkl_to_json(pkl_path, json_path)

            self.assertTrue(json_path.exists())
            with open(json_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            self.assertEqual(payload, sample_corpus)


if __name__ == "__main__":
    unittest.main()
