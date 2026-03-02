import io
import tempfile
from pathlib import Path

import torch
from torch import nn


def mlp(
    in_channels: int = 2,
    out_channels: int = 3,
    flatten: bool = False,
) -> nn.Module:
    layers: list[nn.Module] = []
    if flatten:
        layers.append(nn.Flatten())
    layers += [
        nn.Linear(in_channels, 5),
        nn.ReLU(),
        nn.Linear(5, out_channels),
    ]
    return nn.Sequential(*layers)


def _export_onnx(
    in_channels: int,
    out_channels: int,
    input_shape: tuple[int, ...] | None,
) -> bytes:
    """Export a small MLP to ONNX and return the raw bytes."""
    flatten = input_shape is not None
    model = mlp(in_channels, out_channels, flatten=flatten)

    if input_shape is not None:
        dummy_input = torch.randn(1, *input_shape)
    else:
        dummy_input = torch.randn(1, in_channels)

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

        f.seek(0)
        return f.read()


def inmemory_model(
    in_channels: int = 2,
    out_channels: int = 3,
    input_shape: tuple[int, ...] | None = None,
) -> io.BytesIO:
    """Return an ONNX model as an in-memory buffer."""
    return io.BytesIO(_export_onnx(in_channels, out_channels, input_shape))


def tempfile_model(
    in_channels: int = 2,
    out_channels: int = 3,
    input_shape: tuple[int, ...] | None = None,
    name: str | None = None,
) -> Path:
    """Write an ONNX model to a temp file and return its path.

    If *name* is given the file is written as ``<name>.onnx`` inside a
    temporary directory, so ``path.name`` matches what the server expects.
    """
    data = _export_onnx(in_channels, out_channels, input_shape)

    if name is not None:
        path = Path(tempfile.mkdtemp()) / f"{name}.onnx"
        path.write_bytes(data)
    else:
        f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        f.write(data)
        f.close()
        path = Path(f.name)

    return path
