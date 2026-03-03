import mlclient
import numpy as np


def main():
    with mlclient.MLClient("http://localhost:8000") as client:
        models = client.list_models()

        if not models:
            print("No registered models.")
            exit(0)

        model = models[0]
        input_shape = model.get("input_shape", "")

        shape = tuple([int(size) for size in input_shape[1:-1].split(",")])
        input = np.random.rand(*shape)
        input = np.expand_dims(input, axis=0)

        result = client.infer(model.get("id", 0), input)

        print(result.shape)
        print(result)


if __name__ == "__main__":
    main()
