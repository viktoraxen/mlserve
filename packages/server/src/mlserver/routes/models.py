from fastapi import APIRouter

import mlserver.config as config

router = APIRouter()


@router.get("/models")
async def get_models():
    return [file.stem for file in config.models_path.glob("*.onnx")]
