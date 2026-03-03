from mlclient import MLClient


def test_client_delete_model(client: MLClient, registered_model: int):
    result = client.delete_model(registered_model)

    assert result == registered_model

    models = client.list_models()
    ids = [m["id"] for m in models]
    assert registered_model not in ids
