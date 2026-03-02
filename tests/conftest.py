from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlserve.main import app, configure


@pytest.fixture()
def tmp_dirs(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    db_path = tmp_path / "test.db"
    return models_dir, db_path


@pytest.fixture()
def client(tmp_dirs):
    models_dir, db_path = tmp_dirs
    configure(models_dir=models_dir, db_path=db_path)

    with TestClient(app) as c:
        yield c

    configure(models_dir=models_dir, db_path=db_path)
