import requests

BASE_URL = "http://localhost:8000"


def test_models_endpoint_returns_list():
    response = requests.get(f"{BASE_URL}/models", timeout=2)
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)


def test_infer_returns_valid_output():
    models = requests.get(f"{BASE_URL}/models", timeout=2).json()
    model_name = models[0]

    with open("/home/viktor/Pictures/0001.jpg", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/infer",
            params={"model_name": model_name},
            files={"input": f},
            timeout=2,
        )

    assert response.status_code == 200
