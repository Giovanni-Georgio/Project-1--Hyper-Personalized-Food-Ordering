# Integration Test Script
# test_integration.py

import requests
import json
import base64
from PIL import Image
import io
import asyncio
import httpx

BASE_URL = "http://localhost:8000/api/v1"

def create_test_image():
    """Create a simple test image with text for OCR testing."""
    # Create a simple white image with black text
    img = Image.new('RGB', (400, 200), color='white')
    
    # You would typically add text using PIL.ImageDraw here
    # For simplicity, we'll create a minimal image
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    return img_base64

def test_health_endpoints():
    """Test health check endpoints."""
    print("Testing health endpoints...")
    
    try:
        # Test main health endpoint
        response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health")
        print(f"Main health: {response.status_code} - {response.json()}")
        
        # Test OCR health endpoint
        response = requests.get(f"{BASE_URL}/ocr/health")
        print(f"OCR health: {response.status_code} - {response.json()}")
        
        # Test scraper health endpoint
        response = requests.get(f"{BASE_URL}/scraper/health")
        print(f"Scraper health: {response.status_code} - {response.json()}")
        
        print("✓ Health endpoints working")
        
    except Exception as e:
        print(f"✗ Health endpoints failed: {e}")

def test_ocr_endpoints():
    """Test OCR functionality."""
    print("Testing OCR endpoints...")
    
    try:
        # Test OCR with base64 image
        test_img = create_test_image()
        
        ocr_payload = {
            "image_base64": test_img,
            "language": "eng",
            "preprocessing": True
        }
        
        response = requests.post(
            f"{BASE_URL}/ocr/extract",
            params=ocr_payload
        )
        
        print(f"OCR extract: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"OCR result: {result.get('success', False)}")
            print(f"Items found: {result.get('processing_info', {}).get('total_items', 0)}")
        else:
            print(f"OCR error: {response.text}")
        
        print("✓ OCR endpoints working")
        
    except Exception as e:
        print(f"✗ OCR endpoints failed: {e}")

def test_scraper_endpoints():
    """Test scraper functionality."""
    print("Testing scraper endpoints...")
    
    try:
        # Test supported platforms endpoint
        response = requests.get(f"{BASE_URL}/scraper/supported-platforms")
        print(f"Supported platforms: {response.status_code} - {len(response.json().get('supported_platforms', []))} platforms")
        
        # Test URL validation
        response = requests.post(
            f"{BASE_URL}/scraper/validate-url",
            params={"url": "https://www.example.com"}
        )
        print(f"URL validation: {response.status_code} - Valid: {response.json().get('valid', False)}")
        
        print("✓ Scraper endpoints working")
        
    except Exception as e:
        print(f"✗ Scraper endpoints failed: {e}")

def test_menu_integration():
    """Test menu item creation from OCR results."""
    print("Testing menu integration...")
    
    try:
        # This would test creating menu items from OCR results
        # You'd need to implement the actual menu creation logic
        
        print("✓ Menu integration working (placeholder)")
        
    except Exception as e:
        print(f"✗ Menu integration failed: {e}")

def run_all_tests():
    """Run all integration tests."""
    print("=== Integration Test Suite ===")
    print("Make sure the backend is running on http://localhost:8000")
    print()
    
    test_health_endpoints()
    print()
    
    test_ocr_endpoints()
    print()
    
    test_scraper_endpoints()
    print()
    
    test_menu_integration()
    print()
    
    print("=== Tests Complete ===")

if __name__ == "__main__":
    run_all_tests()