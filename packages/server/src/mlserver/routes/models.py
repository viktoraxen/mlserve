from fastapi import APIRouter
from sqlmodel import Session, select

from mlserver.models.registered_model import RegisteredModel
from mlserver.state import get_engine

router = APIRouter()


@router.get("/models")
async def get_models() -> list[str]:
    with Session(get_engine()) as session:
        models = session.exec(select(RegisteredModel.name)).all()

    return list(models)
