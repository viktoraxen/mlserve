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
            try:
                model = client.pick_model()
            except Exception as e:
                print(f"Could not pick model: {e}")
                raise typer.Exit(1)

            if model is None:
                print("No models available.")
                raise typer.Exit(0)

            input = np.random.rand(*model.input_shape)
            input = np.expand_dims(input, axis=0)

            try:
                result = client.infer(input, model.id)
            except Exception as e:
                print(f"Failed to infer using model with id '{model.id}': {e}")
                raise typer.Exit(1)

            pprint(result.shape)
            pprint(result)
    except httpx.HTTPError:
        print(f"Could not connect to server on URL {url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
