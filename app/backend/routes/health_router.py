from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("")
async def get_health_status():
    """Checks the 'health' of the API."""
    return JSONResponse(content={"status": "API is healthy and running!"})