import io

import numpy as np
from fastapi import UploadFile
from PIL import Image


async def uploadfile_to_ndarray(file: UploadFile) -> np.ndarray:
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    # HWC -> CHW, normalize to [0, 1] float32
    return np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
