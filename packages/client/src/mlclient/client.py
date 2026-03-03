from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np


class MLClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self._client = httpx.Client(base_url=base_url)

    def __enter__(self) -> MLClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def register_onnx_model(
        self,
        name: str,
        model_path: str | Path,
        description: str | None = None,
    ) -> int | None:
        import json

        path = Path(model_path)
        metadata: dict[str, Any] = {"name": name}
        if description is not None:
            metadata["description"] = description

        with open(path, "rb") as f:
            response = self._client.post(
                "/register",
                data={"data": json.dumps(metadata)},
                files={"model": (path.name, f, "application/octet-stream")},
            )

        response.raise_for_status()

        return response.json().get("id")

    def register_pytorch_model(
        self,
        name: str,
        model: Any,
        input_shape: tuple[int, ...],
        description: str | None = None,
    ) -> int | None:
        import contextlib
        import io
        import logging
        import tempfile
        import warnings

        import torch

        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                f"Expected `model` to be of type `torch.nn.Module`, was `{type(model).__name__}`"
            )

        model.eval()
        dummy_input = torch.randn(1, *input_shape)

        with (
            tempfile.NamedTemporaryFile(suffix=".onnx") as f,
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            logging.disable(logging.CRITICAL)
            program = torch.onnx.export(
                model,
                (dummy_input,),
                None,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            logging.disable(logging.NOTSET)

            assert program is not None, "Failed to export model!"
            program.save(f.name)

            return self.register_onnx_model(name, f.name, description=description)

    def delete_model(self, model_id: int) -> int | None:
        response = self._client.post(
            "/delete",
            params={"model_id": model_id},
        )

        response.raise_for_status()
        return response.json().get("id")

    def infer(
        self,
        model_id: int,
        input: Any,
    ) -> np.ndarray:
        import io

        import numpy as np

        if not isinstance(input, np.ndarray):
            import torch

            if isinstance(input, torch.Tensor):
                input = input.numpy()
            else:
                raise TypeError(
                    f"Expected numpy.ndarray or torch.Tensor, got {type(input).__name__}"
                )

        buf = io.BytesIO()
        np.save(buf, input)
        buf.seek(0)

        resp = self._client.post(
            "/infer",
            params={"model_id": model_id},
            files={"input": ("input", buf)},
        )

        resp.raise_for_status()
        result = resp.json()

        assert isinstance(result, list), "Expected inference result to be list."

        return np.array(result)

    def list_models(self) -> list[dict[str, Any]]:
        resp = self._client.get("/models")

        resp.raise_for_status()
        return resp.json()
