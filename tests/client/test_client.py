import numpy as np
from mlclient import MLClient


def test_client_instantiates():
    client = MLClient("http://localhost:8000")
    assert client is not None
    client.close()


def test_client_context_manager():
    with MLClient("http://localhost:8000") as client:
        assert client is not None


def test_client_list_models(client: MLClient):
    models = client.list_models()
    assert isinstance(models, list)


def test_client_register_model(registered_model: int):
    assert isinstance(registered_model, int)


def test_client_infer(client: MLClient, registered_model: int):
    image = np.random.rand(1, 3, 4, 4).astype(np.float32)

    output = client.infer(registered_model, image)

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 10)  # batch of 1, 10 output classes
