import os
from pathlib import Path

import typer
import uvicorn

import mlserver.config as config

app = typer.Typer()


@app.command()
def main(
    host: str = "0.0.0.0",
    port: int = 8000,
    protocol: str = "http",
    db_path: Path = Path(os.environ.get("MLSERVE_DB_PATH", str(Path.home() / ".mlserve" / "database.db"))),
):
    config.configure(
        server_host=host,
        server_port=port,
        server_protocol=protocol,
        db_path=db_path,
    )

    uvicorn.run(
        "mlserver.main:app",
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    app()
