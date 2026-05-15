"""Smoke test for provider_identify_resolution_poc.py.

Runs the script as a subprocess and checks that:
  - the process exits with code 0 (no unhandled exceptions)
  - three expected section headers appear in stdout:
      IDENTITY RESOLUTION RESULTS, FEATURE IMPORTANCE, GOLDEN RECORDS

PYTHONIOENCODING is forced to utf-8 to avoid cp1252 errors on Windows
when the script prints Unicode characters (e.g. arrows, special chars).
"""
import subprocess
import sys
import os
from pathlib import Path
import unittest


class ProviderIdentityResolutionScriptTest(unittest.TestCase):
    def test_script_executes_and_outputs_sections(self):
        # Locate the script relative to this test file (one level up from tests/)
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "provider_identify_resolution_poc.py"

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            # Force UTF-8 so Unicode output doesn't crash on Windows cp1252
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )

        # A non-zero exit code means an unhandled exception was raised
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"Script exited with code {proc.returncode}. stderr:\n{proc.stderr}",
        )

        # Confirm the three main output sections are present and that at least
        # one AUTO-MATCH decision was produced (the known matching providers).
        stdout = proc.stdout
        self.assertIn("IDENTITY RESOLUTION RESULTS", stdout)
        self.assertIn("FEATURE IMPORTANCE", stdout)
        self.assertIn("GOLDEN RECORDS", stdout)
        self.assertIn(
            "AUTO-MATCH",
            stdout,
            msg="Expected at least one AUTO-MATCH decision in pipeline output",
        )


if __name__ == "__main__":
    unittest.main()
