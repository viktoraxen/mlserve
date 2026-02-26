from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile

from mlserve.utils.convert import uploadfile_to_tensor

app = FastAPI()
models_path = Path("/models/onnx/")

_sessions: dict[str, ort.InferenceSession] = {}


def get_session(model_path: Path) -> ort.InferenceSession:
    key = str(model_path)
    if key not in _sessions:
        _sessions[key] = ort.InferenceSession(model_path)
    return _sessions[key]


@app.post("/infer")
async def infer_model(model_name: str, input: UploadFile):
    model_path = models_path / f"{model_name}.onnx"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist.")

    tensor = await uploadfile_to_tensor(input)
    tensor = tensor.unsqueeze(dim=0)

    session = get_session(model_path)
    output = session.run(None, {"input": tensor.numpy()})

    assert isinstance(output, list), "Expected inference output to be a list."
    assert isinstance(output[0], np.ndarray)

    return output[0].tolist()


@app.get("/models")
async def get_models():
    return [file.stem for file in models_path.glob("*.onnx")]
