import httpx
import mlclient
import numpy as np
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

            input = np.random.rand(*model.input_shape)
            input = np.expand_dims(input, axis=0)

            result = client.infer(input, model.id)

            pprint(result.shape)
            pprint(result)
    except httpx.HTTPError:
        print(f"Could not connect to server on URL {url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
