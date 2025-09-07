# OCR API endpoints
# agentic_food_backend/api/ocr.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List
from typing import Dict, Any

from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import menu_items

import base64
import io
from PIL import Image
import logging

from agentic_food_backend.services.ocr_service import ocr_service
from agentic_food_backend.db.schemas import MenuItemCreate
from agentic_food_backend.utils.ocr_helpers import format_menu_for_backend, validate_url
from agentic_food_backend.services.logging_service import log_interaction

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest", status_code=status.HTTP_201_CREATED, tags=["OCR"])
async def ingest_ocr_menu(
    restaurant_id: int,
    menu_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Ingest parsed OCR menu items into the database.
    - **restaurant_id**: target restaurant
    - **menu_data**: output from /ocr/extract
    """
    sections = menu_data.get("sections", {})
    for section in sections:
        for item in section.get("items", []):
            menu_item = MenuItemCreate(
                name=item["name"],
                description=item.get("description", ""),
                price=item["price"],
                currency=item.get("currency", "USD"),
                category=item.get("category"),
                availability=item.get("availability", True),
            )
            # Insert asynchronously
            background_tasks.add_task(
                database.execute,
                menu_items.insert().values(
                    restaurant_id=restaurant_id,
                    **menu_item.dict()
                )
            )
    return {"message": "OCR menu items ingestion scheduled"}

@router.post("/extract", status_code=status.HTTP_200_OK)
async def extract_menu_text(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    language: str = "eng",
    preprocessing: bool = True,
    user_id: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Extract menu text from an image using OCR.
    
    - **image_url**: URL of the menu image
    - **image_base64**: Base64 encoded image data
    - **language**: OCR language (default: eng)
    - **preprocessing**: Apply image preprocessing (default: true)
    - **user_id**: Optional user ID for logging
    """
    if not image_url and not image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either image_url or image_base64 must be provided"
        )
    
    if image_url and not validate_url(image_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image URL format"
        )
    
    try:
        # Process image with OCR
        result = await ocr_service.process_image(
            image_url=image_url,
            image_base64=image_base64,
            language=language,
            preprocessing=preprocessing
        )
        
        if not result['extracted_text']:
            logger.warning("OCR extraction returned empty text")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "No text could be extracted from the image",
                    "warnings": result.get('warnings', [])
                }
            )
        
        # Format result for backend compatibility
        formatted_result = format_menu_for_backend(result)
        
        # Log OCR interaction
        if background_tasks and user_id:
            background_tasks.add_task(
                log_interaction,
                user_id=user_id,
                session_id=f"ocr_{user_id}_{result.get('processing_time', 0)}",
                interaction_type="ocr_extraction",
                query_text=image_url or "base64_image",
                response_data={
                    "items_extracted": result['total_items'],
                    "confidence": result['confidence'],
                    "processing_time": result['processing_time']
                },
                interaction_metadata={
                    "language": language,
                    "preprocessing": preprocessing
                }
            )
        
        return {
            "success": True,
            "extracted_text": result['extracted_text'],
            "confidence": result['confidence'],
            "menu_data": formatted_result,
            "processing_info": {
                "processing_time": result['processing_time'],
                "total_items": result['total_items'],
                "warnings": result.get('warnings', [])
            }
        }
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )

@router.post("/extract/file", status_code=status.HTTP_200_OK)
async def extract_from_uploaded_file(
    file: UploadFile = File(...),
    language: str = Form("eng"),
    preprocessing: bool = Form(True),
    user_id: Optional[int] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Extract menu text from an uploaded image file.
    
    - **file**: Image file upload
    - **language**: OCR language (default: eng)
    - **preprocessing**: Apply image preprocessing (default: true)
    - **user_id**: Optional user ID for logging
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Check file size (10MB limit)
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size too large. Maximum 10MB allowed."
        )
    
    try:
        # Convert uploaded file to base64
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Process with OCR
        result = await ocr_service.process_image(
            image_base64=image_base64,
            language=language,
            preprocessing=preprocessing
        )
        
        if not result['extracted_text']:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "No text could be extracted from the image",
                    "warnings": result.get('warnings', [])
                }
            )
        
        # Format result
        formatted_result = format_menu_for_backend(result)
        
        # Log interaction
        if background_tasks and user_id:
            background_tasks.add_task(
                log_interaction,
                user_id=user_id,
                session_id=f"ocr_file_{user_id}_{result.get('processing_time', 0)}",
                interaction_type="ocr_file_upload",
                query_text=file.filename,
                response_data={
                    "items_extracted": result['total_items'],
                    "confidence": result['confidence'],
                    "file_size": file_size
                },
                interaction_metadata={
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "language": language
                }
            )
        
        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": result['extracted_text'],
            "confidence": result['confidence'],
            "menu_data": formatted_result,
            "processing_info": {
                "processing_time": result['processing_time'],
                "total_items": result['total_items'],
                "warnings": result.get('warnings', [])
            }
        }
        
    except Exception as e:
        logger.error(f"File OCR processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )

@router.post("/batch-extract", status_code=status.HTTP_200_OK)
async def batch_extract_menu_text(
    image_sources: List[dict],
    language: str = "eng",
    preprocessing: bool = True,
    user_id: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Extract menu text from multiple images in batch.
    
    - **image_sources**: List of image sources with 'url' or 'base64' keys
    - **language**: OCR language (default: eng)
    - **preprocessing**: Apply image preprocessing (default: true)
    - **user_id**: Optional user ID for logging
    """
    if not image_sources or len(image_sources) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image source must be provided"
        )
    
    if len(image_sources) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images allowed in batch processing"
        )
    
    # Validate image sources
    for i, source in enumerate(image_sources):
        if not source.get('url') and not source.get('base64'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image source {i+1} must have either 'url' or 'base64' key"
            )
        
        if source.get('url') and not validate_url(source['url']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid URL format in image source {i+1}"
            )
    
    try:
        # Process images in batch
        results = await ocr_service.batch_process_images(
            image_sources=image_sources,
            language=language,
            preprocessing=preprocessing
        )
        
        # Format results
        formatted_results = []
        total_items = 0
        total_processing_time = 0
        
        for result in results:
            formatted_result = format_menu_for_backend(result)
            formatted_results.append({
                "source_index": result['source_index'],
                "success": bool(result['extracted_text']),
                "extracted_text": result['extracted_text'],
                "confidence": result['confidence'],
                "menu_data": formatted_result,
                "processing_info": {
                    "processing_time": result['processing_time'],
                    "total_items": result['total_items'],
                    "warnings": result.get('warnings', [])
                }
            })
            
            total_items += result['total_items']
            total_processing_time += result['processing_time']
        
        # Log batch processing
        if background_tasks and user_id:
            background_tasks.add_task(
                log_interaction,
                user_id=user_id,
                session_id=f"ocr_batch_{user_id}_{total_processing_time}",
                interaction_type="ocr_batch_processing",
                response_data={
                    "images_processed": len(image_sources),
                    "total_items_extracted": total_items,
                    "average_processing_time": total_processing_time / len(results) if results else 0
                },
                interaction_metadata={
                    "batch_size": len(image_sources),
                    "language": language,
                    "preprocessing": preprocessing
                }
            )
        
        successful_extractions = sum(1 for r in formatted_results if r['success'])
        
        return {
            "success": True,
            "results": formatted_results,
            "summary": {
                "total_images": len(image_sources),
                "successful_extractions": successful_extractions,
                "failed_extractions": len(image_sources) - successful_extractions,
                "total_items_extracted": total_items,
                "total_processing_time": total_processing_time
            }
        }
        
    except Exception as e:
        logger.error(f"Batch OCR processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )

@router.get("/health", status_code=status.HTTP_200_OK)
async def ocr_health_check():
    """Check OCR service health and dependencies."""
    try:
        # Test OCR functionality with a simple image
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        result = await ocr_service.process_image(
            image_base64=test_image_b64,
            preprocessing=False
        )
        
        return {
            "status": "healthy",
            "ocr_service": "operational",
            "tesseract_available": bool(result is not None),
            "supported_languages": [ocr_service.language],
            "configuration": {
                "language": ocr_service.language,
                "config": ocr_service.config
            }
        }
        
    except Exception as e:
        logger.error(f"OCR health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "ocr_service": "failed"
            }
        )