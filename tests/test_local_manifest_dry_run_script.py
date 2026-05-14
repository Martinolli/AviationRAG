import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "manifest" / "write-local-sample-manifest.py"
OUTPUT_PATH = ROOT / "data" / "manifest" / "documents.jsonl"


class TestLocalManifestDryRunScript(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists())

    def test_script_writes_ignored_valid_jsonl_and_cleans_up(self):
        if OUTPUT_PATH.exists():
            self.skipTest("Local manifest already exists; refusing to overwrite private local data.")

        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(OUTPUT_PATH.exists())
            self.assertIn("Records loaded: 5", result.stdout)
            self.assertIn("Output is ignored/local-only", result.stdout)

            ignored = subprocess.run(
                ["git", "check-ignore", "data/manifest/documents.jsonl"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(ignored.returncode, 0)
            self.assertIn("data/manifest/documents.jsonl", ignored.stdout)

            lines = [line for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), 5)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(parsed[0]["document_id"], "doc_sample_faa_ac_001")
            self.assertTrue(all(item["metadata"]["fixture"] for item in parsed))
        finally:
            if OUTPUT_PATH.exists():
                OUTPUT_PATH.unlink()


if __name__ == "__main__":
    unittest.main()
