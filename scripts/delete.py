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
            model = client.pick_model()

            if model is None:
                print("No models available.")
                raise typer.Exit(0)

            pprint(client.delete_model(model.id))
    except httpx.HTTPError:
        print(f"Could not connect to server on URL {url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
