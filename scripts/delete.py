import mlclient


def main():
    with mlclient.MLClient("http://localhost:8000") as client:
        models = client.list_models()

        for m in models:
            client.delete_model(m.get("id", 0))


if __name__ == "__main__":
    main()
