import numpy as np
import pytest
import torch
from mlclient import MLClient


def test_client_infer(client: MLClient, registered_model: int):
    image = np.random.rand(1, 3, 4, 4).astype(np.float32)

    output = client.infer(registered_model, image)

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 10)  # batch of 1, 10 output classes


def test_client_infer_with_torch_tensor(client: MLClient, registered_model: int):
    image = torch.randn(1, 3, 4, 4, dtype=torch.float32)

    output = client.infer(registered_model, image)

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 10)


def test_client_infer_rejects_invalid_type(client: MLClient, registered_model: int):
    with pytest.raises(TypeError, match="numpy.ndarray or torch.Tensor"):
        client.infer(registered_model, [1, 2, 3])
