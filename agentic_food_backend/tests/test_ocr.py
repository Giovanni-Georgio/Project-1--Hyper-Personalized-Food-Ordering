# Test script to validate the fixed OCR service
# test_fixed_ocr.py

import asyncio
import base64
from agentic_food_backend.services.ocr_service import ocr_service

async def test_ocr():
    """Test the fixed OCR service with sample data."""
    
    print("Testing Fixed OCR Service...")
    print("=" * 50)
    
    # Test 1: Simple base64 image (1x1 pixel test)
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    try:
        result = await ocr_service.process_image(
            image_base64=test_image_b64,
            preprocessing=False
        )
        
        print("Test 1 - Simple image:")
        print(f"  Success: {result is not None}")
        print(f"  Extracted text: '{result.get('extracted_text', '')}'")
        print(f"  Total items: {result.get('total_items', 0)}")
        print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
        print()
        
    except Exception as e:
        print(f"Test 1 failed: {e}")
        print()
    
    # Test 2: Sample menu text parsing
    sample_text = """
    APPETIZERS
    Caesar Salad 250
    Chicken Wings 300
    MAIN COURSE
    Grilled Salmon 450
    Pasta Carbonara 380
    DESSERTS
    Chocolate Cake 180
    """
    
    print("Test 2 - Text parsing:")
    items = ocr_service.parse_menu_text(sample_text)
    print(f"  Parsed items: {len(items)}")
    for item in items:
        print(f"    - {item['name']}: ${item['price']}")
    print()
    
    # Test 3: Health check
    print("Test 3 - Service health:")
    print(f"  Tesseract config: {ocr_service.config}")
    print(f"  Language: {ocr_service.language}")
    print(f"  Executor: {ocr_service.executor is not None}")
    print()
    
    print("Testing completed!")

if __name__ == "__main__":
    asyncio.run(test_ocr())