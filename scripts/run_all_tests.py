"""Run the repository test suite with the active Python interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run pytest from the repository root and propagate its exit status."""
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
