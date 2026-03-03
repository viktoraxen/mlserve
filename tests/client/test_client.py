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
