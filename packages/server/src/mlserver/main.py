from contextlib import asynccontextmanager

from fastapi import FastAPI
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlmodel import Session, select

import mlserver.config as config
from mlserver.models.registered_model import RegisteredModel
from mlserver.routes import router
from mlserver.state import get_sql_engine


@asynccontextmanager
async def lifespan(application: FastAPI):
    console = Console()

    with Session(get_sql_engine()) as session:
        model_count = len(session.exec(select(RegisteredModel)).all())

    base_url = f"{config.protocol}://{config.host}:{config.port}"

    endpoints = []
    for route in application.routes:
        if not hasattr(route, "methods"):
            continue

        for method in sorted(route.methods):  # type: ignore
            if any([s in route.path for s in ("openapi", "docs", "redoc")]):  # type: ignore
                continue

            endpoints.append(f"{method:<5} {route.path}")  # type: ignore

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    table.add_row("Serving at:", base_url)
    table.add_row("API docs:", f"{base_url}/docs")
    table.add_row("Registered models:", str(model_count))
    table.add_row()

    for i, endpoint in enumerate(endpoints):
        table.add_row("Endpoints:" if i == 0 else "", endpoint)

    console.print(
        Panel(
            Align.center(table),
            title="MLServer",
            width=100,
            expand=True,
            border_style="green",
        )
    )

    yield

    console.print()
    console.print(
        Panel(
            Text("Server stopped.", justify="center"),
            title="MLServer",
            width=100,
            expand=True,
            border_style="red",
        )
    )


app = FastAPI(lifespan=lifespan)
app.include_router(router)
