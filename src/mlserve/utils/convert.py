import io

import torch
from fastapi import UploadFile
from PIL import Image
from torchvision.transforms import v2 as T


async def uploadfile_to_tensor(file: UploadFile) -> torch.Tensor:
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    transform = T.ToTensor()

    return transform(image)
