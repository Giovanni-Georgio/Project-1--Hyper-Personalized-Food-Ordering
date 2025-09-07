# Debugging OCR Endpoint for Testing
# agentic_food_backend/api/debug_ocr.py

from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional
import logging

from agentic_food_backend.services.ocr_service import ocr_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/debug-extract", status_code=status.HTTP_200_OK)
async def debug_extract_menu_text(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    language: str = "eng",
    preprocessing: bool = True
):
    """
    Debug OCR extraction with detailed logging and raw output.
    
    This endpoint provides more detailed information for debugging OCR issues.
    """
    if not image_url and not image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either image_url or image_base64 must be provided"
        )
    
    try:
        # Process image with OCR
        result = await ocr_service.process_image(
            image_url=image_url,
            image_base64=image_base64,
            language=language,
            preprocessing=preprocessing
        )
        
        # Get raw OCR data for debugging
        if image_url:
            image = ocr_service._load_image_from_url(image_url)
        else:
            image = ocr_service._load_image_from_base64(image_base64)
        
        if image and preprocessing:
            processed_image = ocr_service._preprocess_image(image)
        else:
            processed_image = image
        
        # Get detailed OCR data
        detailed_ocr = ocr_service._extract_text_with_confidence(processed_image) if processed_image else {}
        
        # Split text into lines for analysis
        lines = [line.strip() for line in result['extracted_text'].split('\n') if line.strip()]
        
        # Analyze each line
        line_analysis = []
        for i, line in enumerate(lines):
            analysis = {
                "line_number": i + 1,
                "text": line,
                "length": len(line),
                "is_section_header": ocr_service._is_section_header_improved(line),
                "is_noise": ocr_service._is_noise_line(line),
                "extracted_price": None
            }
            
            # Try to extract price
            try:
                from agentic_food_backend.utils.ocr_helpers import extract_price
                price = extract_price(line)
                analysis["extracted_price"] = price
            except:
                pass
            
            line_analysis.append(analysis)
        
        return {
            "success": True,
            "raw_extracted_text": result['extracted_text'],
            "confidence": result['confidence'],
            "processing_time": result['processing_time'],
            "total_lines": len(lines),
            "line_analysis": line_analysis,
            "parsed_sections": result['sections'],
            "total_parsed_items": result['total_items'],
            "warnings": result.get('warnings', []),
            "word_data": detailed_ocr.get('word_data', [])[:20],  # First 20 words only
            "ocr_config": {
                "language": language,
                "preprocessing": preprocessing,
                "tesseract_config": ocr_service.config
            }
        }
        
    except Exception as e:
        logger.error(f"Debug OCR extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug OCR processing failed: {str(e)}"
        )

@router.get("/test-patterns", status_code=status.HTTP_200_OK)
async def test_price_patterns(
    test_text: str = Query(..., description="Text to test price patterns against")
):
    """Test price extraction patterns against sample text."""
    try:
        from agentic_food_backend.utils.ocr_helpers import extract_price
        from agentic_food_backend.config import PRICE_PATTERNS
        
        results = {
            "input_text": test_text,
            "extracted_price": extract_price(test_text),
            "pattern_matches": []
        }
        
        # Test each pattern individually
        import re
        for i, pattern in enumerate(PRICE_PATTERNS):
            match = re.search(pattern, test_text)
            results["pattern_matches"].append({
                "pattern_index": i,
                "pattern": pattern,
                "matched": bool(match),
                "match_text": match.group() if match else None
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Pattern testing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pattern testing failed: {str(e)}"
        )