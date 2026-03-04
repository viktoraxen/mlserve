import httpx
import mlclient
import typer
from torch import nn


def main(
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
):
    model = nn.Sequential(
        nn.Linear(10, 15),
        nn.ReLU(),
        nn.Linear(15, 20),
    )
    url = f"{protocol}://{host}:{port}"

    try:
        with mlclient.MLClient(url) as client:
            client.register_pytorch_model(
                name="Multi-layer perceptron",
                description="Simple two-layer mlp.",
                model=model,
                input_shape=(10,),
            )
    except httpx.HTTPError:
        print(f"Could not connect to server on URL {url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
