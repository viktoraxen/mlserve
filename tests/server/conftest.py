import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mlserver.config import configure
from mlserver.main import app
from mlserver.state import reset as reset_state

from tests.utils.models import inmemory_model


@pytest.fixture()
def tmp_dirs(tmp_path: Path) -> tuple[Path, Path]:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    db_path = tmp_path / "test.db"
    return models_dir, db_path


@pytest.fixture()
def client(tmp_dirs: tuple[Path, Path]) -> Generator[TestClient]:
    models_dir, db_path = tmp_dirs
    configure(models_dir=models_dir, db_path=db_path)
    reset_state()

    with TestClient(app) as c:
        yield c

    configure(models_dir=models_dir, db_path=db_path)
    reset_state()


@pytest.fixture()
def registered_model(client: TestClient) -> int:
    """Register a tiny model and return its id."""
    model_file = inmemory_model(3 * 4 * 4, 10, input_shape=(3, 4, 4))
    name = "test_model"

    resp = client.post(
        "/register",
        data={"data": json.dumps({"name": name})},
        files={"model": (f"{name}.onnx", model_file, "application/octet-stream")},
    )
    assert resp.status_code == 200
    return resp.json()["id"]
