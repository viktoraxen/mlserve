from pathlib import Path


def count_parameters(model_path: Path) -> int:
    import numpy as np
    import onnx

    model = onnx.load(str(model_path))
    return sum(int(np.prod(t.dims)) for t in model.graph.initializer)
