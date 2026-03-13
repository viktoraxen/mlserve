import json

from fastapi import APIRouter, Form, HTTPException, UploadFile
from sqlmodel import Session

import mlserver.config as config
from mlserver.logger import logger
from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_sql_engine
from mlserver.utils.onnx import get_model_info

router = APIRouter()


@router.post("/register")
async def register_model(model: UploadFile, data: str = Form()):
    metadata = json.loads(data)
    model_name = metadata.get("name")

    # TODO: Validate input data

    logger.info(f"Registering model '{model_name}'.")

    if model.filename is None:
        logger.error(f"Model '{model_name}' did not provide filename.")

        raise HTTPException(status_code=400, detail="Model filename is required!")

    model_path = config.models_path / model.filename

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            f.write(model.file.read())
    except Exception as e:
        logger.error(f"Model '{model_name}' could not be written to path '{model_path}'!")

        raise HTTPException(status_code=500, detail=f"Writing model file failed: {e}")

    try:
        model_info = get_model_info(model_path)
    except Exception as e:
        logger.error(f"Model '{model_name}' provided invalid ONNX model!")

        model_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid ONNX model: {e}")

    try:
        with Session(get_sql_engine()) as session:
            registered_model = RegisteredModel(
                name=metadata["name"],
                path=str(model_path),
                description=metadata.get("description"),
                **model_info,
            )
            session.add(registered_model)
            session.commit()
            session.refresh(registered_model)
    except Exception as e:
        logger.error(f"Model '{model_name}' could not be written to database!")

        model_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Writing to database failed: {e}")

    return registered_model
