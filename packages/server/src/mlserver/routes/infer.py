import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from sqlmodel import Session, select

from mlserver.logger import logger
from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_onnx_session, get_sql_engine
from mlserver.utils.convert import uploadfile_to_ndarray

router = APIRouter()


@router.post("/infer")
async def infer_model(model_id: int, input: UploadFile):
    if input.content_type not in (None, "application/octet-stream", "application/x-npy"):
        logger.warning(
            f"Failed to run inferrence using model with id '{model_id}': Unsupported content type: {input.content_type}"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Failed to run inferrence using model with id '{model_id}': Unsupported content type: {input.content_type}",
        )

    try:
        array = await uploadfile_to_ndarray(input)
    except Exception as e:
        logger.warning(
            f"Failed to run inferrence using model with id '{model_id}', could not convert input data: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Failed to run inferrence using model with id '{model_id}', could not convert input data: {e}",
        )

    if array.ndim == 0:
        logger.warning(
            f"Failed to run inferrence using model with id '{model_id}': Input array must have at least one dimension."
        )

        raise HTTPException(
            status_code=400,
            detail="Failed to run inferrence using model with id '{model_id}': Input array must have at least one dimension.",
        )

    try:
        with Session(get_sql_engine()) as session:
            result = session.exec(
                select(RegisteredModel).where(RegisteredModel.id == model_id)
            ).first()
    except Exception as e:
        logger.error(f"Failed to find model with id '{model_id}': {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to find model with id '{model_id}': {e}",
        )

    if not result:
        logger.warning(f"Model with id '{model_id}' does not exist.")

        raise HTTPException(status_code=404, detail=f"Model with id '{model_id}' does not exist.")

    model_path = result.path

    try:
        session = get_onnx_session(model_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: array})
    except Exception as e:
        logger.error(f"Failed to run inference using model with id '{model_id}': {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to run inference using model with id '{model_id}': {e}",
        )

    if not isinstance(output, list) or not isinstance(output[0], np.ndarray):
        logger.error(f"Unexpected inference output type using model '{model_id}': {type(output)}")

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected inference output type using model '{model_id}': {type(output)}",
        )

    logger.info(f"Inferred using model with id '{model_id}'. {array.shape} -> {output[0].shape}.")

    return output[0].tolist()
