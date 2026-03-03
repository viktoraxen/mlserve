import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from sqlmodel import Session, select

from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_onnx_session, get_sql_engine
from mlserver.utils.convert import uploadfile_to_ndarray

router = APIRouter()


@router.post("/infer")
async def infer_model(model_id: int, input: UploadFile):
    try:
        with Session(get_sql_engine()) as session:
            result = session.exec(
                select(RegisteredModel).where(RegisteredModel.id == model_id)
            ).first()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to find model with id '{model_id}'. Exception {e}"
        )

    if not result:
        raise HTTPException(status_code=404, detail=f"Model with id '{model_id}' does not exist.")

    model_path = result.path

    array = await uploadfile_to_ndarray(input)

    session = get_onnx_session(model_path)
    output = session.run(None, {"input": array})

    assert isinstance(output, list), "Expected inference output to be a list."
    assert isinstance(output[0], np.ndarray), "Inference output contains no ndarray."

    return output[0].tolist()
