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
    if model.content_type not in (None, "application/octet-stream", "application/x-onnx"):
        logger.warning(f"Failed to register model: Unsupported content type: {model.content_type}")

        raise HTTPException(
            status_code=400,
            detail=f"Failed to register model: Unsupported content type: {model.content_type}",
        )

    if not data:
        logger.warning("Failed to register model: No metadata provided.")

        raise HTTPException(
            status_code=400,
            detail="Failed to register model: No metadata provided.",
        )

    try:
        metadata = json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to register model: Invalid JSON metadata: {e}")

        raise HTTPException(
            status_code=400,
            detail=f"Failed to register model: Invalid JSON metadata: {e}",
        )

    model_name = metadata.get("name")

    if not model_name:
        logger.warning("Failed to register model: Missing 'name' in metadata.")

        raise HTTPException(
            status_code=400,
            detail="Failed to register model: Missing 'name' in metadata.",
        )

    if model.filename is None:
        logger.warning(f"Failed to register model '{model_name}': No filename provided.")

        raise HTTPException(
            status_code=400,
            detail=f"Failed to register model '{model_name}': No filename provided.",
        )

    model_path = config.models_path / model.filename

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            f.write(model.file.read())
    except Exception as e:
        logger.error(f"Model '{model_name}' could not be written to path '{model_path}': {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Model '{model_name}' could not be written to path '{model_path}': {e}",
        )

    try:
        model_info = get_model_info(model_path)
    except Exception as e:
        logger.warning(f"Failed to register model '{model_name}', invalid ONNX model: {e}")

        model_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to register model '{model_name}', invalid ONNX model: {e}",
        )

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
        model_path.unlink(missing_ok=True)

        logger.error(f"Model '{model_name}' could not be written to database: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Model '{model_name}' could not be written to database: {e}",
        )

    logger.info(f"Registered model '{model_name}' to id '{registered_model.id}'.")

    return registered_model
