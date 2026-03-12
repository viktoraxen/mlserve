import io
import json

import numpy as np

from tests.utils.models import inmemory_model_custom_input_name


def test_models_endpoint_returns_list(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()

    assert isinstance(models, list)


def test_infer_returns_valid_output(client, registered_model):
    array = np.random.rand(1, 3, 4, 4).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)

    response = client.post(
        "/infer",
        params={"model_id": registered_model},
        files={"input": ("input", buf)},
    )

    assert response.status_code == 200

    output = response.json()

    assert isinstance(output, list)
    assert len(output) == 1  # batch of 1
    assert len(output[0]) == 10  # 10 output classes


def test_infer_with_custom_input_name(client):
    model_file = inmemory_model_custom_input_name(
        in_channels=4, out_channels=3, input_name="x"
    )

    resp = client.post(
        "/register",
        data={"data": json.dumps({"name": "custom_input"})},
        files={"model": ("custom_input.onnx", model_file, "application/octet-stream")},
    )
    assert resp.status_code == 200
    model_id = resp.json()["id"]

    array = np.random.rand(1, 4).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)

    response = client.post(
        "/infer",
        params={"model_id": model_id},
        files={"input": ("input", buf)},
    )

    assert response.status_code == 200
    output = response.json()
    assert isinstance(output, list)
    assert len(output) == 1
    assert len(output[0]) == 3


def test_infer_nonexistent_model_returns_404(client):
    array = np.random.rand(1, 3, 4, 4).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)

    response = client.post(
        "/infer",
        params={"model_id": 999999},
        files={"input": ("input", buf)},
    )

    assert response.status_code == 404
