from mlclient import MLClient


def test_client_delete_model(client: MLClient, registered_model: int):
    deleted_id = client.delete_model(registered_model).id

    assert deleted_id == registered_model

    models = client.models()
    ids = [m.id for m in models]

    assert registered_model not in ids
