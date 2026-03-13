import os
from pathlib import Path

_default_dir = Path.home() / ".mlserve"

models_path = Path(os.environ.get("MLSERVE_MODELS_PATH", str(_default_dir / "models")))
sqlite_url = "sqlite:///" + os.environ.get("MLSERVE_DB_PATH", str(_default_dir / "database.db"))


host = "0.0.0.0"
port = 8000
protocol = "http"


def configure(
    *,
    models_dir: Path | None = None,
    db_path: Path | None = None,
    server_host: str | None = None,
    server_port: int | None = None,
    server_protocol: str | None = None,
) -> None:
    global models_path, sqlite_url, host, port, protocol

    if models_dir is not None:
        models_path = models_dir

    if db_path is not None:
        sqlite_url = f"sqlite:///{db_path}"

    if server_host is not None:
        host = server_host

    if server_port is not None:
        port = server_port

    if server_protocol is not None:
        protocol = server_protocol
