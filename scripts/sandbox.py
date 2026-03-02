"""
Sandbox server that uses temporary directories instead of production paths.

    uv run python scripts/sandbox.py

Runs on http://localhost:9000 with an isolated models dir and database.
"""

import tempfile
from pathlib import Path

import uvicorn

from mlserver.main import app, configure

tmp = tempfile.mkdtemp(prefix="mlserve_sandbox_")
models_dir = Path(tmp) / "models"
models_dir.mkdir()
db_path = Path(tmp) / "sandbox.db"

configure(models_dir=models_dir, db_path=db_path)

print(f"Sandbox models dir: {models_dir}")
print(f"Sandbox database:   {db_path}")


def main():
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
    main()
