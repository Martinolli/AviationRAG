import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_FILES_DIR = ROOT / "src" / "scripts" / "py_files"
sys.path.insert(0, str(PY_FILES_DIR))

import config  # noqa: E402


class TestConfigPaths(unittest.TestCase):
    def test_project_root_detection(self):
        self.assertEqual(config.PROJECT_ROOT, ROOT)
        self.assertTrue((config.PROJECT_ROOT / "README.md").exists())

    def test_expected_subpaths(self):
        self.assertEqual(config.PY_SCRIPTS_DIR, ROOT / "src" / "scripts" / "py_files")
        self.assertEqual(config.JS_SCRIPTS_DIR, ROOT / "src" / "scripts" / "js_files")
        self.assertEqual(config.DATA_DIR, ROOT / "data")
        self.assertEqual(config.PKL_FILENAME, ROOT / "data" / "raw" / "aviation_corpus.pkl")
        self.assertEqual(config.EMBEDDINGS_FILE, ROOT / "data" / "embeddings" / "aviation_embeddings.json")


if __name__ == "__main__":
    unittest.main()
