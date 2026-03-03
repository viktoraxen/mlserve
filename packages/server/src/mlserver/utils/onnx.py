from pathlib import Path


def _get_shape(tensor_type) -> list[int]:
    shape = []

    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            continue
        shape.append(dim.dim_value)

    return shape


def get_model_info(model_path: Path) -> dict:
    import numpy as np
    import onnx

    model = onnx.load(str(model_path))

    num_parameters = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    input_shape = _get_shape(model.graph.input[0].type.tensor_type)
    output_shape = _get_shape(model.graph.output[0].type.tensor_type)

    return {
        "num_parameters": num_parameters,
        "input_shape": str(input_shape),
        "output_shape": str(output_shape),
    }
