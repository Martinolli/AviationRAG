import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER_PATH = ROOT / "src" / "scripts" / "py_files" / "aviationrag_manager.py"


class TestManagerHelp(unittest.TestCase):
    def test_manager_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(MANAGER_PATH), "--help"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("AviationRAG Processing Pipeline", result.stdout)


if __name__ == "__main__":
    unittest.main()
