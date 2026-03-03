import mlclient
import numpy as np

from tests.utils.data import parse_shape


def main():
    with mlclient.MLClient("http://localhost:8000") as client:
        models = client.list_models()
        model = models[0]
        input_shape = model.get("input_shape", "")

        input = np.random.rand(*parse_shape(input_shape))
        input = np.expand_dims(input, axis=0)

        result = client.infer(model.get("id", 0), input)

        print(result.shape)
        print(result)


if __name__ == "__main__":
    main()
