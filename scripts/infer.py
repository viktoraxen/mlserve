import mlclient


def main():
    with mlclient.MLClient("http://localhost:8000") as client:
        models = client.list_models()
        model = models[0]
        input_shape = model.get("input_shape")
        print(input_shape)

        # client.infer(
        #     model.get("id", 0),
        # )


if __name__ == "__main__":
    main()
