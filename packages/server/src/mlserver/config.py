import os
from pathlib import Path

models_path = Path(os.environ.get("MLSERVE_MODELS_PATH", "/models"))
sqlite_url = "sqlite:///" + os.environ.get("MLSERVE_DB_PATH", "/models/database.db")


def configure(*, models_dir: Path | None = None, db_path: Path | None = None) -> None:
    global models_path, sqlite_url

    if models_dir is not None:
        models_path = models_dir

    if db_path is not None:
        sqlite_url = f"sqlite:///{db_path}"
