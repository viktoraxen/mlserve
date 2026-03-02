from mlclient import MLClient

from tests.utils.data import tempfile_image


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


def test_client_register_model(registered_model: str):
    assert registered_model == "test_model"


def test_client_infer(client: MLClient, registered_model: str):
    image_path = tempfile_image()

    output = client.infer(registered_model, image_path)

    assert isinstance(output, list)
    assert len(output) == 1  # batch of 1
    assert len(output[0]) == 10  # 10 output classes
