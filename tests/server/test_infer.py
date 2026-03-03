import io

import numpy as np


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
