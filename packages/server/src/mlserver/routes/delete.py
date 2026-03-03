from fastapi import APIRouter, HTTPException
from sqlmodel import Session, delete

from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_sql_engine

router = APIRouter()


@router.post("/delete")
async def delete_model(model_id: str):
    try:
        with Session(get_sql_engine()) as session:
            session.exec(delete(RegisteredModel).where(RegisteredModel.id == model_id))  # type: ignore[arg-type]
            session.commit()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Deleting model with id '{model_id}' failed with exception {e}"
        )

    return {"message": f"Model '{id}' deleted successfully."}
