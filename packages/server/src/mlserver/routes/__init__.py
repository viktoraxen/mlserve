from fastapi import APIRouter

from mlserver.routes.delete import router as delete_router
from mlserver.routes.infer import router as infer_router
from mlserver.routes.models import router as models_router
from mlserver.routes.register import router as register_router

router = APIRouter()
router.include_router(delete_router)
router.include_router(infer_router)
router.include_router(models_router)
router.include_router(register_router)
