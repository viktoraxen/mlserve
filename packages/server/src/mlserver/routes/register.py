import json

from fastapi import APIRouter, Form, HTTPException, UploadFile
from sqlmodel import Session

import mlserver.config as config
from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_engine
from mlserver.utils.onnx import count_parameters

router = APIRouter()


@router.post("/register")
async def register_model(model: UploadFile, data: str = Form()):
    if model.filename is None:
        raise HTTPException(status_code=500, detail="Model filename is required!")

    metadata = json.loads(data)
    model_path = config.models_path / model.filename

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(model.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Writing model file failed: {e}")

    num_parameters = count_parameters(model_path)

    try:
        with Session(get_engine()) as session:
            session.add(
                RegisteredModel(
                    name=metadata["name"],
                    path=str(model_path),
                    description=metadata.get("description"),
                    num_parameters=num_parameters,
                )
            )

            session.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Writing to database failed: {e}")

    return {"message": f"Model '{metadata['name']}' registered successfully."}
