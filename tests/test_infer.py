import io
import json
import tempfile

import torch
from PIL import Image


def _register_infer_model(client):
    """Export and register a tiny model that accepts (1, 3, 4, 4) input."""
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 10),
    )

    dummy_input = torch.randn(1, 3, 4, 4)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        program = torch.onnx.export(
            model,
            (dummy_input,),
            None,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        assert program is not None
        program.save(f.name)

        metadata = {"name": "infer_test"}

        f.seek(0)
        resp = client.post(
            "/register",
            data={"data": json.dumps(metadata)},
            files={"model": ("infer_test.onnx", f, "application/octet-stream")},
        )
        assert resp.status_code == 200


def _make_synthetic_image() -> io.BytesIO:
    """Create a 4x4 RGB PNG image in memory."""
    img = Image.new("RGB", (4, 4), color=(127, 64, 200))
    buf = io.BytesIO()

    img.save(buf, format="PNG")
    buf.seek(0)

    return buf


def test_models_endpoint_returns_list(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()

    assert isinstance(models, list)


def test_infer_returns_valid_output(client):
    _register_infer_model(client)

    image_buf = _make_synthetic_image()

    response = client.post(
        "/infer",
        params={"model_name": "infer_test"},
        files={"input": ("test.png", image_buf, "image/png")},
    )

    assert response.status_code == 200

    output = response.json()

    assert isinstance(output, list)
    assert len(output) == 1  # batch of 1
    assert len(output[0]) == 10  # 10 output classes
