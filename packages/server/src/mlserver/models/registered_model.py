from datetime import datetime

from pydantic import field_serializer
from sqlmodel import Field, SQLModel


class RegisteredModel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    path: str = Field(exclude=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    num_parameters: int

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%d %H:%M:%S")
