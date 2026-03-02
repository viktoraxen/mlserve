import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Form, HTTPException, UploadFile
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from mlserve.models.registered_model import RegisteredModel
from mlserve.utils.convert import uploadfile_to_ndarray

app = FastAPI()
models_path = Path(os.environ.get("MLSERVE_MODELS_PATH", "/models/mlserve/onnx/"))
sqlite_url = "sqlite:///" + os.environ.get("MLSERVE_DB_PATH", "/db/mlserve/database.db")

_sessions: dict[str, ort.InferenceSession] = {}
_engine: Engine | None = None


def configure(*, models_dir: Path | None = None, db_path: Path | None = None):
    global models_path, sqlite_url, _engine, _sessions

    if models_dir is not None:
        models_path = models_dir

    if db_path is not None:
        sqlite_url = f"sqlite:///{db_path}"

    _engine = None
    _sessions = {}


def get_session(model_path: Path) -> ort.InferenceSession:
    key = str(model_path)

    if key not in _sessions:
        _sessions[key] = ort.InferenceSession(model_path)

    return _sessions[key]


def get_engine() -> Engine:
    global _engine

    if _engine is not None:
        return _engine

    _engine = create_engine(sqlite_url, echo=True)
    SQLModel.metadata.create_all(_engine)

    return _engine


@app.post("/infer")
async def infer_model(model_name: str, input: UploadFile):
    model_path = models_path / f"{model_name}.onnx"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist.")

    array = await uploadfile_to_ndarray(input)
    array = np.expand_dims(array, axis=0)

    session = get_session(model_path)
    output = session.run(None, {"input": array})

    assert isinstance(output, list), "Expected inference output to be a list."
    assert isinstance(output[0], np.ndarray)

    return output[0].tolist()


@app.post("/register")
async def register_model(model: UploadFile, data: str = Form()):
    if model.filename is None:
        raise HTTPException(status_code=500, detail="Model filename is required!")

    metadata = json.loads(data)
    model_path = models_path / model.filename

    try:
        with Session(get_engine()) as session:
            session.add(
                RegisteredModel(
                    name=metadata["name"],
                    path=str(model_path),
                )
            )

            session.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Writing to database failed: {e}")

    try:
        with open(model_path, "wb") as f:
            f.write(model.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Writing model file failed: {e}")

    return {"message": f"Model '{metadata['name']}' registered successfully."}


@app.get("/models")
async def get_models():
    return [file.stem for file in models_path.glob("*.onnx")]
