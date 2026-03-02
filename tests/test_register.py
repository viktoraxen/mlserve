import json
import tempfile

import torch


def test_register_model(client):
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 15),
    )

    dummy_input = torch.randn(1, 10)

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

        metadata = {"name": "Test MLP"}

        f.seek(0)
        response = client.post(
            "/register",
            data={"data": json.dumps(metadata)},
            files={"model": ("test_mlp.onnx", f, "application/octet-stream")},
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}!"
