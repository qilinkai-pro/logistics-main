"""Run data generation, training and test evaluation in sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    run("generate_data.py")
    run("train.py")
    run("test.py")
    run("infer.py")


if __name__ == "__main__":
    main()
