import json

from fastapi.testclient import TestClient

from tests.utils.data import parse_shape
from tests.utils.models import inmemory_model


def test_register_model(registered_model):
    assert isinstance(registered_model, int)


def test_register_model_with_description(client: TestClient):
    model_file = inmemory_model(3 * 4 * 4, 10, input_shape=(3, 4, 4))
    name = "described_model"
    description = "A test model with a description"

    resp = client.post(
        "/register",
        data={"data": json.dumps({"name": name, "description": description})},
        files={"model": (f"{name}.onnx", model_file, "application/octet-stream")},
    )

    assert resp.status_code == 200

    models = client.get("/models").json()
    match = [m for m in models if m["id"] == resp.json()["id"]]
    assert len(match) == 1
    assert match[0]["description"] == description


def test_register_model_stores_metadata(client: TestClient, registered_model: int):
    models = client.get("/models").json()
    match = [m for m in models if m["id"] == registered_model]

    assert len(match) == 1

    model = match[0]

    assert model["num_parameters"] > 0
    assert len(parse_shape(model["input_shape"])) > 0
    assert len(parse_shape(model["output_shape"])) > 0


def test_register_model_missing_filename_returns_error(client: TestClient):
    model_file = inmemory_model()

    resp = client.post(
        "/register",
        data={"data": json.dumps({"name": "bad_model"})},
        files={"model": (None, model_file, "application/octet-stream")},
    )

    assert resp.status_code != 200
