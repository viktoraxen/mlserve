import json

from mlclient.client import MLClient


def main():
    with MLClient("http://localhost:8000") as client:
        models = client.list_models()
        print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()
