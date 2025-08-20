from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from typing import Optional
import numpy as np


from app.models.schemas import SearchResponse, ErrorResponse, ProductCategory
from app.services.ml\_service import ml\_service
from app.services.search\_service import search\_service
from app.core.config import settings


*# Configure logging*
logger = logging.getLogger(\_\_name\_\_)


*# Create router*
router = APIRouter()
*# AIzaSyBvejC\_\_4bNQ4qf1kc9C5ooORaayyWYw24*


def validate\_image\_file(file: UploadFile) -> Image.Image:
    """
    Validate and process uploaded image file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        PIL Image object
        
    Raises:
        HTTPException: If file validation fails
    """
    try:
        *# Check file size*
        if file.size and file.size > settings.MAX\_FILE\_SIZE:
            raise HTTPException(
                status\_code=413,
                detail=f"File size {file.size} bytes exceeds maximum allowed size of {settings.MAX\_FILE\_SIZE} bytes"
            )
        
        *# Check file type - be more flexible with content type detection*
        if file.content\_type:
            *# Normalize content type and check if it's an image*
            content\_type = file.content\_type.lower()
            if not any(img\_type in content\_type for img\_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']):
                raise HTTPException(
                    status\_code=400,
                    detail=f"File type {file.content\_type} not allowed. Allowed types: {settings.ALLOWED\_IMAGE\_TYPES}"
                )
        else:
            *# If no content type, check file extension*
            if file.filename:
                file\_ext = file.filename.lower().split('.')[-1]
                if file\_ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                    raise HTTPException(
                        status\_code=400,
                        detail=f"File extension .{file\_ext} not allowed. Allowed extensions: jpg, jpeg, png, gif, bmp, tiff"
                    )
            else:
                logger.warning("No content type or filename detected, but continuing with processing")
        
        *# Read and validate image*
        image\_data = file.file.read()
        if not image\_data:
            raise HTTPException(status\_code=400, detail="Empty file uploaded")
        
        logger.info(f"Image file details - Size: {len(image\_data)} bytes, Content-Type: {file.content\_type}, Filename: {file.filename}")
        
        *# Convert to PIL Image*
        image = Image.open(io.BytesIO(image\_data))
        
        logger.info(f"PIL Image details - Format: {image.format}, Mode: {image.mode}, Size: {image.size}")
        
        *# Validate image format - be more flexible with format detection*
        if hasattr(image, 'format') and image.format:
            *# Convert to uppercase for comparison and handle common variations*
            detected\_format = image.format.upper()
            if detected\_format not in ['JPEG', 'JPG', 'PNG', 'GIF', 'BMP', 'TIFF']:
                logger.warning(f"Unusual image format detected: {detected\_format}")
        else:
            logger.warning("No image format detected, but continuing with processing")
        
        *# Convert to RGB if necessary (CLIP expects RGB)*
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Image validated successfully: {image.size} {image.mode}")
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating image file: {e}")
        raise HTTPException(status\_code=500, detail="Error processing image file")



@router.post("/search", response\_model=SearchResponse)
async def search\_products(
    image: Optional[UploadFile] = File(None, description="Product image for visual search"),
    description: str = Form("", description="Text description of the product"),
    category: Optional[ProductCategory] = Form(None, description="Optional product category filter")
):
    """
    Search for products using image and/or text description.
    
    This endpoint performs multimodal semantic search using:
    - CLIP embeddings for image understanding
    - Sentence Transformers for text understanding
    - Hybrid search combining both modalities
    
    Returns top 5 product recommendations sorted by similarity.
    """
    try:
        logger.info(f"Search request received - Image: {image.filename if image else 'None'}, "
                   f"Description: '{description[:50]}...', Category: {category}")
        
        *# Validate that at least one search input is provided*
        if not image and not description.strip():
            raise HTTPException(
                status\_code=400,
                detail="At least one of image or description must be provided"
            )
        
        *# Process image if provided*
        image\_embedding = None
        if image:
            try:
                pil\_image = validate\_image\_file(image)
                image\_embedding = ml\_service.process\_image(pil\_image)
                logger.info("Image processed successfully")
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                raise HTTPException(status\_code=500, detail="Failed to process image")
        
        *# Process text description if provided*
        text\_embedding = None
        if description.strip():
            try:
                text\_embedding = ml\_service.process\_text(description.strip())
                logger.info("Text description processed successfully")
            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                raise HTTPException(status\_code=500, detail="Failed to process text description")
        
        *# Perform product search*
        try:
            recommendations = search\_service.search\_products(
                image\_embedding=image\_embedding,
                text\_embedding=text\_embedding,
                category=category
            )
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            raise HTTPException(status\_code=500, detail="Failed to perform product search")
        
        *# Determine search type*
        if image\_embedding is not None and text\_embedding is not None:
            search\_type = "hybrid"
        elif image\_embedding is not None:
            search\_type = "image"
        else:
            search\_type = "text"
        
        *# Generate search metadata*
        metadata = search\_service.get\_search\_metadata(search\_type, len(recommendations))
        
        *# Create response*
        response = SearchResponse(
            query\_type=search\_type,
            total\_results=len(recommendations),
            recommendations=recommendations,
            search\_metadata=metadata
        )
        
        logger.info(f"Search completed successfully. Found {len(recommendations)} recommendations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {e}")
        raise HTTPException(status\_code=500, detail="Internal server error")



@router.get("/health")
async def health\_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy", "service": "Semantic Product Search API"}



@router.get("/info")
async def get\_service\_info():
    """Get service information and configuration."""
    return {
        "service\_name": settings.PROJECT\_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "max\_recommendations": settings.MAX\_RECOMMENDATIONS,
        "similarity\_threshold": settings.SIMILARITY\_THRESHOLD,
        "supported\_image\_types": settings.ALLOWED\_IMAGE\_TYPES,
        "max\_file\_size\_mb": settings.MAX\_FILE\_SIZE / (1024 \* 1024)
    }