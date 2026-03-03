import io
import tempfile
from pathlib import Path

from PIL import Image


def _create_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(127, 64, 200))


def inmemory_image() -> io.BytesIO:
    """Create a 4x4 RGB PNG image in memory."""
    buf = io.BytesIO()
    _create_image().save(buf, format="PNG")
    buf.seek(0)
    return buf


def tempfile_image() -> Path:
    """Create a 4x4 RGB PNG image as a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    _create_image().save(f, format="PNG")
    f.close()
    return Path(f.name)


def parse_shape(shape_str: str) -> tuple:
    """Parse a shape string into a tuple."""
    return tuple([int(size) for size in shape_str[1:-1].split(",")])
