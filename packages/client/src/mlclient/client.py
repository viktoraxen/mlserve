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
            response = self._client.post(
                "/register",
                data={"data": json.dumps({"name": name})},
                files={"model": (path.name, f, "application/octet-stream")},
            )

        response.raise_for_status()
        return response.json()

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

    def register_pytorch_model(
        self,
        name: str,
        model: Any,
        input_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        import tempfile

        import torch

        dummy_input = torch.randn(1, *input_shape)

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            program = torch.onnx.export(
                model,
                (dummy_input,),
                None,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )

            assert program is not None, "Failed to export model!"
            program.save(f.name)

            return self.register_model(name, f.name)

    def list_models(self) -> list[str]:
        resp = self._client.get("/models")
        resp.raise_for_status()
        return resp.json()
