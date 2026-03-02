import mlclient
from torch import nn


def main():
    model = nn.Sequential(
        nn.Linear(10, 15),
        nn.ReLU(),
        nn.Linear(15, 20),
    )

    with mlclient.MLClient("http://localhost:8000") as client:
        client.register_pytorch_model(
            "Multi-layer perceptron",
            model,
            (10,),
        )


if __name__ == "__main__":
    main()
