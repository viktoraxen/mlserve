import json

import httpx
import mlclient
import typer


def main(
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
):
    url = f"{protocol}://{host}:{port}"

    try:
        with mlclient.MLClient(url) as client:
            models = client.list_models()
            print(json.dumps(models, indent=2))
    except httpx.HTTPError:
        print(f"Could not connect to server on URL {url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
