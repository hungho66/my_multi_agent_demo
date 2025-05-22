from fastapi import APIRouter
from .health_router import router as health_check_router
from .agent_router import router as multi_agent_router

api_router = APIRouter()

api_router.include_router(health_check_router, prefix="/health", tags=["Health Checks"])
api_router.include_router(multi_agent_router, prefix="/agents", tags=["Multi-Agent Processing"])