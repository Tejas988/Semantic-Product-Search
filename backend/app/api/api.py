from fastapi import APIRouter
from app.api.endpoints import search

# Create main API router
api_router = APIRouter()

# Include search endpoints
api_router.include_router(search.router, prefix="/search", tags=["search"])

# You can add more endpoint routers here as the project grows
# api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
# api_router.include_router(products.router, prefix="/products", tags=["products"])

