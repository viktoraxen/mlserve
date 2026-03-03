import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

import mlserver.config as config
from mlserver.state import get_onnx_session
from mlserver.utils.convert import uploadfile_to_ndarray

router = APIRouter()


@router.post("/infer")
async def infer_model(model_name: str, input: UploadFile):

    model_path = config.models_path / f"{model_name}.onnx"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist.")

    array = await uploadfile_to_ndarray(input)
    array = np.expand_dims(array, axis=0)

    session = get_onnx_session(model_path)
    output = session.run(None, {"input": array})

    assert isinstance(output, list), "Expected inference output to be a list."
    assert isinstance(output[0], np.ndarray)

    return output[0].tolist()
