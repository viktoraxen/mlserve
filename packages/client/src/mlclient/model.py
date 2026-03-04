from __future__ import annotations

from dataclasses import dataclass


def parse_shape(shape_str: str) -> tuple:
    """Parse a shape string into a tuple."""
    return tuple([int(size) for size in shape_str[1:-1].split(",")])


@dataclass
class Model:
    id: int
    name: str
    description: str
    created_at: str
    num_parameters: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    @staticmethod
    def from_dict(data: dict) -> "Model":
        data["id"] = int(data.get("id", -1))
        data["num_parameters"] = int(data.get("num_parameters", 0))
        data["input_shape"] = parse_shape(data.get("input_shape", (0,)))
        data["output_shape"] = parse_shape(data.get("output_shape", (0,)))

        return Model(**data)
