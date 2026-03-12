# MLServe

A two-package project for storing and inferring ML-models remotely.

The project is divided into a `server` and a `client` package. Both are needed for a full experience, but it is unlikely both need to be installed on the same computer.

## Installation

### UV

```bash
uv add "git+https://github.com/viktoraxen/mlserve.git#subdirectory=packages/{client|server}"
```

### Pip

```bash
pip install "git+https://github.com/viktoraxen/mlserve.git#subdirectory=packages/{client|server}"
```
