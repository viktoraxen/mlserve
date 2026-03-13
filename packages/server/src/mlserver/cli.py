import typer
import uvicorn

import mlserver.config as config

app = typer.Typer()


@app.command()
def main(
    host: str = "0.0.0.0",
    port: int = 8000,
    protocol: str = "http",
):
    config.configure(server_host=host, server_port=port, server_protocol=protocol)

    uvicorn.run(
        "mlserver.main:app",
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    app()
