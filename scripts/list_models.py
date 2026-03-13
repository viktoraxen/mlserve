import httpx
import mlclient
import typer
from rich.pretty import pprint


def main(
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
):
    url = f"{protocol}://{host}:{port}"

    try:
        with mlclient.MLClient(url) as client:
            try:
                models = client.models()
            except Exception as e:
                print(f"Failed to list models: {e}")
                raise typer.Exit(1)

            pprint(models)

    except httpx.HTTPError:
        print(f"Could not connect to server on URL {url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
