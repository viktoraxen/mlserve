from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mlclient import MLClient
from mlserver.config import configure
from mlserver.main import app
from mlserver.state import reset as reset_state

from tests.utils.models import tempfile_model


@pytest.fixture()
def tmp_dirs(tmp_path: Path) -> tuple[Path, Path]:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    db_path = tmp_path / "test.db"
    return models_dir, db_path


@pytest.fixture()
def client(tmp_dirs: tuple[Path, Path]) -> Generator[MLClient]:
    models_dir, db_path = tmp_dirs
    configure(models_dir=models_dir, db_path=db_path)
    reset_state()

    with TestClient(app) as transport:
        c = MLClient()
        c._client = transport
        yield c

    configure(models_dir=models_dir, db_path=db_path)
    reset_state()


@pytest.fixture()
def registered_model(client: MLClient) -> int:
    """Register a tiny model via the client and return its id."""
    name = "test_model"
    model_path = tempfile_model(3 * 4 * 4, 10, input_shape=(3, 4, 4), name=name)
    return client.register_onnx_model(name, model_path)
