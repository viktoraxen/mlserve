from mlclient import MLClient
from mlclient.model import Model


def test_client_delete_model(client: MLClient, registered_model: Model):
    deleted_id = client.delete_model(registered_model.id).id

    assert deleted_id == registered_model.id

    models = client.models()
    ids = [m.id for m in models]

    assert registered_model not in ids
