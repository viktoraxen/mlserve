import httpx
import mlclient
import numpy as np
import typer


def main(
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
):
    url = f"{protocol}://{host}:{port}"

    try:
        with mlclient.MLClient(url) as client:
            model = client.pick_model()
            input_shape = model.get("input_shape")

            shape = tuple([int(size) for size in input_shape[1:-1].split(",")])
            input = np.random.rand(*shape)
            input = np.expand_dims(input, axis=0)

            result = client.infer(input, model.get("id"))

            print(result.shape)
            print(result)
    except httpx.HTTPError:
        print(f"Could not connect to server on URL {url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
