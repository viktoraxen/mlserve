from pathlib import Path

import onnxruntime as ort
from sqlalchemy import Engine
from sqlmodel import SQLModel, create_engine

import mlserver.config as config

_sessions: dict[str, ort.InferenceSession] = {}
_engine: Engine | None = None


def reset() -> None:
    global _engine, _sessions
    _engine = None
    _sessions = {}


def get_onnx_session(model_path: str | Path) -> ort.InferenceSession:
    key = str(model_path)

    if key not in _sessions:
        _sessions[key] = ort.InferenceSession(model_path)

    return _sessions[key]


def get_sql_engine() -> Engine:
    global _engine

    if _engine is not None:
        return _engine

    db_path = Path(config.sqlite_url.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(config.sqlite_url, echo=True)
    SQLModel.metadata.create_all(_engine)

    return _engine
