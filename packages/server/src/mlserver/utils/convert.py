import io

import numpy as np
from fastapi import UploadFile


async def uploadfile_to_ndarray(file: UploadFile) -> np.ndarray:
    content = await file.read()

    return np.load(io.BytesIO(content))
