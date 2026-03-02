import json

from fastapi import APIRouter, Form, HTTPException, UploadFile
from sqlmodel import Session

import mlserver.config as config
from mlserver.state import get_engine
from mlserver.models.registered_model import RegisteredModel

router = APIRouter()


@router.post("/register")
async def register_model(model: UploadFile, data: str = Form()):
    if model.filename is None:
        raise HTTPException(status_code=500, detail="Model filename is required!")

    metadata = json.loads(data)
    model_path = config.models_path / model.filename

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
