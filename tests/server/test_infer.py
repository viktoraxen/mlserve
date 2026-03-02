from tests.utils.data import inmemory_image


def test_models_endpoint_returns_list(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()

    assert isinstance(models, list)


def test_infer_returns_valid_output(client, registered_model):
    image = inmemory_image()

    response = client.post(
        "/infer",
        params={"model_name": registered_model},
        files={"input": ("test.png", image, "image/png")},
    )

    assert response.status_code == 200

    output = response.json()

    assert isinstance(output, list)
    assert len(output) == 1  # batch of 1
    assert len(output[0]) == 10  # 10 output classes
