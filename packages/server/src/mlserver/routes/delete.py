from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, delete, select

from mlserver.logger import logger
from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_sql_engine

router = APIRouter()


@router.post("/delete")
async def delete_model(model_id: int):
    logger.info(f"Deleting model with id '{model_id}'.")

    try:
        with Session(get_sql_engine()) as session:
            existing = session.exec(
                select(RegisteredModel).where(RegisteredModel.id == model_id)
            ).first()

            if not existing:
                logger.warning(f"Model with id '{model_id}' does not exist!")

                raise HTTPException(
                    status_code=404,
                    detail=f"Model with id '{model_id}' does not exist.",
                )

            session.exec(delete(RegisteredModel).where(RegisteredModel.id == model_id))  # type: ignore[arg-type]
            session.commit()
    except HTTPException:  # Catch 404 raised above
        raise
    except Exception as e:
        logger.error(f"Failed to remove model with id '{model_id}' from database!")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove model with id '{model_id}' from database: {e}",
        )

    try:
        Path(existing.path).unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Failed to delete model file for model with id '{model_id}'!")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model file for model with id '{model_id}': {e}",
        )

    return existing
