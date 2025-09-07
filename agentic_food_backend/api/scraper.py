# Web Scraper API endpoints
# agentic_food_backend/api/scraper.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, status
from typing import Optional, Dict, Any
import asyncio, httpx, os
from concurrent.futures import ThreadPoolExecutor
import logging

from agentic_food_backend.utils.ocr_helpers import validate_url, detect_platform
from agentic_food_backend.services.logging_service import log_interaction

logger = logging.getLogger(__name__)
router = APIRouter()
OCR_URL = os.getenv("BASE_API_URL", "http://localhost:8000") + "/api/v1/ocr"

# Thread pool for running synchronous scraping operations
executor = ThreadPoolExecutor(max_workers=4)

def scrape_menu_sync(url: str) -> Dict[str, Any]:
    """
    Synchronous menu scraping function (placeholder).
    This should implement your selenium scraping logic.
    """
    try:
        # Import your actual scraper here
        # from agentic_food_backend.scrapers.selenium_scraper import SeleniumScraper
        
        # For now, return a placeholder response
        platform = detect_platform(url)
        
        # This is where you'd call your actual scraper
        # scraper = SeleniumScraper()
        # result = scraper.scrape_menu(url)
        
        # Placeholder result
        result = {
            "restaurant": {
                "name": "Sample Restaurant",
                "platform": platform,
                "url": url,
                "cuisine_type": "Italian",
                "rating": 4.2,
                "address": "123 Sample Street",
                "phone": "+1-555-0123"
            },
            "sections": [
                {
                    "name": "Appetizers",
                    "items": [
                        {
                            "name": "Caesar Salad",
                            "description": "Fresh romaine lettuce with Caesar dressing",
                            "price": 12.99,
                            "currency": "USD",
                            "availability": True
                        }
                    ]
                }
            ],
            "total_items": 1,
            "total_sections": 1,
            "processing_time": 2.5,
            "warnings": []
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        raise Exception(f"Scraping failed: {str(e)}")

@router.get("/scrape-and-ingest", status_code=status.HTTP_202_ACCEPTED, tags=["Scraper"])
async def scrape_and_ingest(
    restaurant_id: int,
    menu_url: str = Query(..., description="URL of restaurant menu page"),
    background_tasks: BackgroundTasks = None
):
    """
    Full pipeline:
    1. Scrape screenshot via Selenium (omitted here)
    2. Call OCR extract endpoint
    3. Ingest into DB via /ocr/ingest
    """
    # 1. Scrape menu page to image URL (implement your Selenium logic separately)
    screenshot_url = await scrape_menu_to_image_url(menu_url)  # implement this helper
    
    # 2. Call OCR
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OCR_URL}/extract",
            json={"image_url": screenshot_url}
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="OCR extraction failed")
        ocr_payload = resp.json()
    
    # 3. Ingest into DB
    background_tasks.add_task(
        client.post,
        f"{OCR_URL}/ingest",
        json={
            "restaurant_id": restaurant_id,
            "menu_data": ocr_payload["menu_data"]
        }
    )
    return {"message": "Scrape→OCR→Ingest pipeline started"}

@router.get("/menu", status_code=status.HTTP_200_OK)
async def scrape_restaurant_menu(
    url: str = Query(..., description="URL of the restaurant menu page"),
    user_id: Optional[int] = Query(None, description="Optional user ID for logging"),
    background_tasks: BackgroundTasks = None
):
    """
    Scrape menu data from a restaurant website.
    
    - **url**: Restaurant menu page URL
    - **user_id**: Optional user ID for interaction logging
    """
    if not validate_url(url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL format"
        )
    
    platform = detect_platform(url)
    
    try:
        # Run the synchronous scraping in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, scrape_menu_sync, url)
        
        # Log scraping interaction
        if background_tasks and user_id:
            background_tasks.add_task(
                log_interaction,
                user_id=user_id,
                session_id=f"scrape_{user_id}_{result.get('processing_time', 0)}",
                interaction_type="menu_scraping",
                query_text=url,
                response_data={
                    "items_scraped": result.get('total_items', 0),
                    "sections_scraped": result.get('total_sections', 0),
                    "processing_time": result.get('processing_time', 0)
                },
                interaction_metadata={
                    "platform": platform,
                    "restaurant_name": result.get('restaurant', {}).get('name', 'Unknown')
                }
            )
        
        return {
            "success": True,
            "platform": platform,
            "menu_data": result,
            "scraping_info": {
                "url": url,
                "platform_detected": platform,
                "processing_time": result.get('processing_time', 0),
                "warnings": result.get('warnings', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Menu scraping failed for {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scraping failed: {str(e)}"
        )

@router.get("/restaurant-info", status_code=status.HTTP_200_OK)
async def get_restaurant_info(
    url: str = Query(..., description="URL of the restaurant page"),
    user_id: Optional[int] = Query(None, description="Optional user ID for logging")
):
    """
    Extract basic restaurant information from a URL.
    
    - **url**: Restaurant page URL
    - **user_id**: Optional user ID for logging
    """
    if not validate_url(url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL format"
        )
    
    platform = detect_platform(url)
    
    try:
        # This would implement restaurant info extraction
        # For now, return placeholder data
        restaurant_info = {
            "name": "Sample Restaurant",
            "platform": platform,
            "url": url,
            "cuisine_type": "Italian",
            "rating": 4.2,
            "address": "123 Sample Street",
            "phone": "+1-555-0123",
            "delivery_time": "30-45 min",
            "minimum_order": 15.00,
            "delivery_fee": 2.99
        }
        
        return {
            "success": True,
            "restaurant": restaurant_info,
            "platform": platform
        }
        
    except Exception as e:
        logger.error(f"Restaurant info extraction failed for {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Restaurant info extraction failed: {str(e)}"
        )

@router.get("/supported-platforms", status_code=status.HTTP_200_OK)
async def get_supported_platforms():
    """Get list of supported restaurant platforms."""
    from agentic_food_backend.config import SUPPORTED_PLATFORMS
    
    return {
        "supported_platforms": SUPPORTED_PLATFORMS,
        "total_platforms": len(SUPPORTED_PLATFORMS),
        "description": "List of restaurant platforms that can be scraped"
    }

@router.post("/validate-url", status_code=status.HTTP_200_OK)
async def validate_restaurant_url(url: str):
    """
    Validate if a URL is from a supported platform and is accessible.
    
    - **url**: URL to validate
    """
    if not validate_url(url):
        return {
            "valid": False,
            "reason": "Invalid URL format",
            "platform": None
        }
    
    platform = detect_platform(url)
    
    try:
        # Basic URL accessibility check
        import requests
        from agentic_food_backend.config import USER_AGENT
        
        headers = {'User-Agent': USER_AGENT}
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        
        accessible = response.status_code < 400
        
        return {
            "valid": accessible,
            "platform": platform,
            "status_code": response.status_code,
            "accessible": accessible,
            "supported_platform": platform != "generic"
        }
        
    except Exception as e:
        logger.warning(f"URL validation failed for {url}: {e}")
        return {
            "valid": False,
            "reason": f"URL not accessible: {str(e)}",
            "platform": platform
        }

@router.get("/health", status_code=status.HTTP_200_OK)
async def scraper_health_check():
    """Check scraper service health."""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "scraper_service": "operational",
            "selenium_available": True,  # You'd check this properly
            "supported_platforms_count": len(SUPPORTED_PLATFORMS) if 'SUPPORTED_PLATFORMS' in globals() else 0
        }
        
    except Exception as e:
        logger.error(f"Scraper health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "scraper_service": "failed"
        }