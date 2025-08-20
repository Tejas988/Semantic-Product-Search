from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ProductCategory(str, Enum):
    """Product categories for classification."""
    CLOTHING = "clothing"
    SHOES = "shoes"
    ELECTRONICS = "electronics"
    HOME = "home"
    BEAUTY = "beauty"
    SPORTS = "sports"
    AUTOMOTIVE = "automotive"
    OTHER = "other"


class SearchRequest(BaseModel):
    """Request model for product search."""
    description: Optional[str] = Field(
        default="",
        description="Text description of the product to search for",
        max_length=1000
    )
    category: Optional[ProductCategory] = Field(
        default=None,
        description="Optional product category to narrow down search"
    )


class ProductRecommendation(BaseModel):
    """Model for individual product recommendation."""
    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    category: ProductCategory = Field(..., description="Product category")
    brand: Optional[str] = Field(None, description="Product brand")
    price: Optional[float] = Field(None, description="Product price")
    image_url: Optional[str] = Field(None, description="Product image URL")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    match_type: str = Field(..., description="Type of match (image, text, or hybrid)")


class SearchResponse(BaseModel):
    """Response model for product search results."""
    query_type: str = Field(..., description="Type of search performed")
    total_results: int = Field(..., description="Total number of results found")
    recommendations: List[ProductRecommendation] = Field(..., description="List of product recommendations")
    search_metadata: dict = Field(..., description="Additional search metadata")
    alignment_check: Optional[dict] = Field(None, description="Image-description alignment validation results")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")

