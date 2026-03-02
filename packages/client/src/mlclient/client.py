from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class MLClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self._client = httpx.Client(base_url=base_url)

    def __enter__(self) -> MLClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def register_model(
        self,
        name: str,
        model_path: str | Path,
    ) -> dict[str, Any]:
        import json

        path = Path(model_path)

        with open(path, "rb") as f:
            resp = self._client.post(
                "/register",
                data={"data": json.dumps({"name": name})},
                files={"model": (path.name, f, "application/octet-stream")},
            )

        resp.raise_for_status()
        return resp.json()

    def infer(
        self,
        model_name: str,
        input_path: str | Path,
    ) -> Any:
        path = Path(input_path)

        with open(path, "rb") as f:
            resp = self._client.post(
                "/infer",
                params={"model_name": model_name},
                files={"input": (path.name, f)},
            )

        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> list[str]:
        resp = self._client.get("/models")
        resp.raise_for_status()
        return resp.json()
