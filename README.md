# MLServe

A two-package project for storing and inferring ML-models remotely.

The project is divided into a `mlserver` and a `mlclient` package. Both are needed for a full experience, but it is unlikely both need to be installed on the same computer.

## Installation

```bash
# UV
uv add "git+https://github.com/viktoraxen/mlserve.git#subdirectory=packages/{client|server}"

# Pip
pip install "git+https://github.com/viktoraxen/mlserve.git#subdirectory=packages/{client|server}"
```

## Server

### Starting the server

```bash
# With defaults (http://0.0.0.0:8000)
uv run server

# Custom adress
uv run server --host 127.0.0.1 --port 3000 --protocol https
```

The server stores ONNX model files and a SQLite database on disk.

### Configuration

When running outside of Docker, set these environment variables to control where data is stored:

| Variable | Default | Description |
|---|---|---|
| `MLSERVE_MODELS_PATH` | `~/.mlserve/models` | Directory for ONNX model files |
| `MLSERVE_DB_PATH` | `~/.mlserve/database.db` | SQLite database file path |

The defaults store everything under `~/.mlserve/`. The Docker image overrides these to `/models/onnx` and `/models/database.db`. To use a custom location:

```bash
MLSERVE_MODELS_PATH=./data/onnx MLSERVE_DB_PATH=./data/database.db uv run server
```

The database and model directory are created automatically on first request, no migrations needed.

## Client

For PyTorch model registration, `torch` is needed and is assumed to already be installed to keep dependencies light.

### Connecting

```python
from mlclient import MLClient

client = MLClient("http://localhost:8000")

# or as a context manager
with MLClient("http://localhost:8000") as client:
    ...
```

### Registering a PyTorch model

```python
import torch.nn as nn
from mlclient import MLClient

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

with MLClient("http://localhost:8000") as client:
    model_id = client.register_pytorch_model(
        name="My MLP",
        model=model,
        input_shape=(10,),           # without batch dimension
        description="Simple two-layer MLP",
    )
```

### Registering a pre-exported ONNX model

```python
with MLClient("http://localhost:8000") as client:
    model_id = client.register_onnx_model(
        name="My model",
        model_path="model.onnx",
    )
```

ONNX models are validated on upload — invalid files are rejected with a `400` error.

### Running inference

```python
import numpy as np
from mlclient import MLClient

with MLClient("http://localhost:8000") as client:
    models = client.models()
    model = models[0]

    input_data = np.random.rand(1, *model.input_shape).astype(np.float32)  # batch dim first
    result = client.infer(input_data, model.id)
    # result is a numpy.ndarray
```

The input must include a batch dimension as the first axis. `model.input_shape` does **not** include the batch dimension, prepend it yourself (e.g. with `np.expand_dims` or by shaping as `(1, *model.input_shape)`).

`infer` also accepts a `torch.Tensor`, which is converted to numpy internally.

### Listing and deleting models

```python
with MLClient("http://localhost:8000") as client:
    for model in client.models():
        print(model.name, model.input_shape, model.output_shape)

    client.delete_model(model_id=1)
```

> [!NOTE]
> `pick_model()`, `delete_model()` without an ID, and `infer()` without an ID launch an interactive fuzzy picker in the terminal. These require a TTY. They won't work in notebooks or non-interactive scripts.
