from fastapi import APIRouter
from sqlmodel import Session, select

from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_sql_engine

router = APIRouter()


@router.get("/models")
async def get_models() -> list[RegisteredModel]:
    with Session(get_sql_engine()) as session:
        models = session.exec(select(RegisteredModel)).all()

    return list(models)
