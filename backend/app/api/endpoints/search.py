from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from typing import Optional
import numpy as np

from app.models.schemas import SearchResponse, ErrorResponse, ProductCategory
from app.services.ml_service import ml_service
from app.services.search_service import search_service
from app.services.alignment_service import alignment_service
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()
# AIzaSyBvejC__4bNQ4qf1kc9C5ooORaayyWYw24

def validate_image_file(file: UploadFile) -> Image.Image:
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
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size {file.size} bytes exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Check file type - be more flexible with content type detection
        if file.content_type:
            # Normalize content type and check if it's an image
            content_type = file.content_type.lower()
            if not any(img_type in content_type for img_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file.content_type} not allowed. Allowed types: {settings.ALLOWED_IMAGE_TYPES}"
                )
        else:
            # If no content type, check file extension
            if file.filename:
                file_ext = file.filename.lower().split('.')[-1]
                if file_ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File extension .{file_ext} not allowed. Allowed extensions: jpg, jpeg, png, gif, bmp, tiff"
                    )
            else:
                logger.warning("No content type or filename detected, but continuing with processing")
        
        # Read and validate image
        image_data = file.file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        logger.info(f"Image file details - Size: {len(image_data)} bytes, Content-Type: {file.content_type}, Filename: {file.filename}")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"PIL Image details - Format: {image.format}, Mode: {image.mode}, Size: {image.size}")
        
        # Validate image format - be more flexible with format detection
        if hasattr(image, 'format') and image.format:
            # Convert to uppercase for comparison and handle common variations
            detected_format = image.format.upper()
            if detected_format not in ['JPEG', 'JPG', 'PNG', 'GIF', 'BMP', 'TIFF']:
                logger.warning(f"Unusual image format detected: {detected_format}")
        else:
            logger.warning("No image format detected, but continuing with processing")
        
        # Convert to RGB if necessary (CLIP expects RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Image validated successfully: {image.size} {image.mode}")
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating image file: {e}")
        raise HTTPException(status_code=500, detail="Error processing image file")


@router.post("/search", response_model=SearchResponse)
async def search_products(
    image: Optional[UploadFile] = File(None, description="Product image for visual search"),
    description: str = Form("", description="Text description of the product"),
    category: Optional[ProductCategory] = Form(None, description="Optional product category filter"),
    check_alignment: bool = Form(False, description="Whether to check if image and description align")
):
    """
    Search for products using image and/or text description.
    
    This endpoint performs multimodal semantic search using:
    - CLIP embeddings for image understanding
    - Sentence Transformers for text understanding
    - Hybrid search combining both modalities
    
    If check_alignment is True and both image and description are provided,
    it will validate that they make sense together before proceeding.
    
    Returns top 5 product recommendations sorted by similarity.
    """
    try:
        logger.info(f"Search request received - Image: {image.filename if image else 'None'}, "
                   f"Description: '{description[:50]}...', Category: {category}, Check Alignment: {check_alignment}")
        
        # Validate that at least one search input is provided
        if not image and not description.strip():
            raise HTTPException(
                status_code=400,
                detail="At least one of image or description must be provided"
            )
        
        # Check alignment if requested and both inputs are provided
        alignment_result = None
        if check_alignment and image and description.strip():
            try:
                pil_image = validate_image_file(image)
                alignment_result = alignment_service.validate_image_description_alignment(pil_image, description)
                
                # If alignment check fails, return warning but continue with search
                if not alignment_result.get("is_aligned", True):
                    logger.warning(f"Image-description alignment check failed: {alignment_result}")
                    # You could choose to return an error here instead of continuing
                    # raise HTTPException(status_code=400, detail="Image and description do not align")
                
                logger.info("Alignment check completed")
            except Exception as e:
                logger.error(f"Alignment check failed: {e}")
                # Continue with search even if alignment check fails
                alignment_result = {"error": str(e)}
        
        # Process image if provided
        image_embedding = None
        if image:
            try:
                pil_image = validate_image_file(image)
                image_embedding = ml_service.process_image(pil_image)
                logger.info("Image processed successfully")
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to process image")
        
        # Process text description if provided
        text_embedding = None
        if description.strip():
            try:
                text_embedding = ml_service.process_text(description.strip())
                logger.info("Text description processed successfully")
            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to process text description")
        
        # Perform product search
        try:
            recommendations = search_service.search_products(
                image_embedding=image_embedding,
                text_embedding=text_embedding,
                category=category
            )
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to perform product search")
        
        # Determine search type
        if image_embedding is not None and text_embedding is not None:
            search_type = "hybrid"
        elif image_embedding is not None:
            search_type = "image"
        else:
            search_type = "text"
        
        # Generate search metadata
        metadata = search_service.get_search_metadata(search_type, len(recommendations))
        
        # Create response
        response = SearchResponse(
            query_type=search_type,
            total_results=len(recommendations),
            recommendations=recommendations,
            search_metadata=metadata
        )
        
        # Add alignment result to response if available
        if alignment_result:
            response.alignment_check = alignment_result
        
        logger.info(f"Search completed successfully. Found {len(recommendations)} recommendations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/validate-alignment")
async def validate_image_description_alignment(
    image: UploadFile = File(..., description="Product image to validate against description"),
    description: str = Form(..., description="Text description to validate against the image")
):
    """
    Validate if the user's description aligns with the image content.
    
    This endpoint helps ensure that the search request makes sense by:
    1. Generating a caption for the uploaded image using CLIP
    2. Using Gemini LLM to check if the description aligns with the image content
    3. Providing feedback and suggestions for better alignment
    
    Returns alignment validation results with confidence scores and reasoning.
    """
    try:
        logger.info(f"Alignment validation request received - Image: {image.filename}, "
                   f"Description: '{description[:50]}...'")
        
        # Validate image file
        try:
            pil_image = validate_image_file(image)
            logger.info("Image validated successfully for alignment check")
        except Exception as e:
            logger.error(f"Image validation failed for alignment: {e}")
            raise HTTPException(status_code=500, detail="Failed to validate image for alignment check")
        
        # Perform alignment validation
        try:
            alignment_result = alignment_service.validate_image_description_alignment(pil_image, description)
            logger.info("Alignment validation completed successfully")
        except Exception as e:
            logger.error(f"Alignment validation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to perform alignment validation")
        
        return {
            "success": True,
            "alignment_result": alignment_result,
            "message": "Alignment validation completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in alignment validation endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy", "service": "Semantic Product Search API"}


@router.get("/info")
async def get_service_info():
    """Get service information and configuration."""
    return {
        "service_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "max_recommendations": settings.MAX_RECOMMENDATIONS,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "supported_image_types": settings.ALLOWED_IMAGE_TYPES,
        "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024)
    }

