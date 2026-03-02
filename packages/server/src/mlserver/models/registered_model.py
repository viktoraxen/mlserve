from sqlmodel import Field, SQLModel


class RegisteredModel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    path: str
