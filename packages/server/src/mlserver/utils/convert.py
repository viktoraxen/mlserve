import io

import numpy as np
from fastapi import UploadFile


async def uploadfile_to_ndarray(file: UploadFile) -> np.ndarray:
    content = await file.read()

    array = np.load(io.BytesIO(content))
    return array.astype(np.float32, copy=False)
