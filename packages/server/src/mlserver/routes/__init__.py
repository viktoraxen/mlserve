from fastapi import APIRouter

from mlserver.routes.infer import router as infer_router
from mlserver.routes.register import router as register_router
from mlserver.routes.models import router as models_router

router = APIRouter()
router.include_router(infer_router)
router.include_router(register_router)
router.include_router(models_router)
