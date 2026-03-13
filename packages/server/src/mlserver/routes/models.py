from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

from mlserver.logger import logger
from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_sql_engine

router = APIRouter()


@router.get("/models")
async def get_models() -> list[RegisteredModel]:
    try:
        with Session(get_sql_engine()) as session:
            models = session.exec(select(RegisteredModel)).all()
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")

        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")

    logger.info(f"Fetched models ({len(models)}).")

    return list(models)
