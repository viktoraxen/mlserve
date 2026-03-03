from fastapi import APIRouter, HTTPException
from sqlmodel import Session, delete, select

from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_sql_engine

router = APIRouter()


@router.post("/delete")
async def delete_model(model_id: int):
    try:
        with Session(get_sql_engine()) as session:
            existing = session.exec(
                select(RegisteredModel).where(RegisteredModel.id == model_id)
            ).first()

            if not existing:
                raise HTTPException(
                    status_code=404, detail=f"Model with id '{model_id}' does not exist."
                )

            session.exec(delete(RegisteredModel).where(RegisteredModel.id == model_id))  # type: ignore[arg-type]
            session.commit()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Deleting model with id '{model_id}' failed with exception {e}"
        )

    return {
        "message": f"Model '{model_id}' deleted successfully.",
        "id": model_id,
    }
