# agentic_food_backend/utils/ocr_helpers.py

import re
from typing import Optional, List, Dict, Any

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.strip()

def extract_price(text: str) -> Optional[float]:
    """Simple price extraction."""
    if not text:
        return None
    
    # Look for price patterns
    price_patterns = [
        r'\$(\d+\.?\d*)',  # $12.99
        r'(\d+\.?\d*)\$',  # 12.99$
        r'(\d+)\s*/-',     # 250 /-
        r'(\d+\.?\d*)',    # Just numbers
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None

def validate_url(url: str) -> bool:
    """Validate URL format."""
    if not url or not isinstance(url, str):
        return False
    
    url_pattern = r'^https?:\/\/.+\..+'
    return bool(re.match(url_pattern, url))

def detect_platform(url: str) -> str:
    """Detect restaurant platform from URL."""
    if not url:
        return "generic"
    
    url_lower = url.lower()
    platforms = ["zomato", "ubereats", "grubhub", "doordash", "swiggy", "foodpanda"]
    
    for platform in platforms:
        if platform in url_lower:
            return platform
    
    return "generic"

def format_menu_for_backend(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format OCR result to match backend menu item schema."""
    formatted_sections = []
    
    for section_name, items in ocr_result.get('sections', {}).items():
        formatted_items = []
        
        for item in items:
            formatted_item = {
                "name": item.get('name', ''),
                "description": item.get('description', ''),
                "price": item.get('price', 0.0),
                "currency": item.get('currency', 'USD'),
                "category": "other",  # Simple default
                "dietary_restrictions": [],
                "availability": item.get('availability', True)
            }
            formatted_items.append(formatted_item)
        
        formatted_sections.append({
            "name": section_name,
            "items": formatted_items
        })
    
    return {
        "sections": formatted_sections,
        "total_items": ocr_result.get('total_items', 0),
        "processing_info": {
            "confidence": ocr_result.get('confidence', 0),
            "processing_time": ocr_result.get('processing_time', 0),
            "warnings": ocr_result.get('warnings', [])
        }
    }