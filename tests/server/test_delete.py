import io

import numpy as np
from fastapi.testclient import TestClient


def test_delete_model(client: TestClient, registered_model: int):
    response = client.post("/delete", params={"model_id": registered_model})

    assert response.status_code == 200
    assert response.json()["id"] == registered_model


def test_delete_model_removes_from_db(client: TestClient, registered_model: int):
    client.post("/delete", params={"model_id": registered_model})

    models = client.get("/models").json()
    ids = [m["id"] for m in models]
    assert registered_model not in ids


def test_delete_nonexistent_model(client: TestClient):
    response = client.post("/delete", params={"model_id": 999999})

    assert response.status_code == 404


def test_delete_then_infer_returns_404(client: TestClient, registered_model: int):
    client.post("/delete", params={"model_id": registered_model})

    array = np.random.rand(1, 3, 4, 4).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)

    response = client.post(
        "/infer",
        params={"model_id": registered_model},
        files={"input": ("input", buf)},
    )

    assert response.status_code == 404
