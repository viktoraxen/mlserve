import pytest
from mlclient import MLClient

from tests.utils.models import mlp, tempfile_model


def test_client_register_model(registered_model: int):
    assert isinstance(registered_model, int)


def test_client_register_pytorch_model(client: MLClient):
    model = mlp(3 * 4 * 4, 10, flatten=True)
    model_id = client.register_pytorch_model("pytorch_model", model, input_shape=(3, 4, 4))

    assert isinstance(model_id, int)


def test_client_register_pytorch_model_rejects_non_module(client: MLClient):
    with pytest.raises(ValueError, match="torch.nn.Module"):
        client.register_pytorch_model("bad", "not_a_module", input_shape=(3,))


def test_client_register_onnx_model_with_description(client: MLClient):
    model_path = tempfile_model(3 * 4 * 4, 10, input_shape=(3, 4, 4), name="desc_model")
    model_id = client.register_onnx_model("desc_model", model_path, description="A described model")

    assert isinstance(model_id, int)

    models = client.models()
    match = [m for m in models if m.id == model_id]

    assert len(match) == 1
    assert match[0].description == "A described model"
