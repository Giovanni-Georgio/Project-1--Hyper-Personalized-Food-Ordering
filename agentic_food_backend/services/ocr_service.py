# Updated OCR Service with improved parsing for your menu format
# agentic_food_backend/services/ocr_service.py

import cv2
import numpy as np
import pytesseract
from PIL import Image
import requests
import base64
import io
import re
import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import logging

from agentic_food_backend.config import (
    TESSERACT_CMD, OCR_LANGUAGE, USER_AGENT
)

logger = logging.getLogger(__name__)

class OCRService:
    """OCR service using your original working tesseract code with improved parsing."""
    
    def __init__(self):
        """Initialize OCR service with original working config."""
        # Set Tesseract command path
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        
        # Use your original working config
        self.config = '--oem 1 --psm 6'  # Exact same as your working code
        self.language = OCR_LANGUAGE or 'eng'
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _load_image_from_url(self, image_url: str) -> Optional[np.ndarray]:
        """Load image from URL and return as cv2 image (BGR)."""
        try:
            headers = {'User-Agent': USER_AGENT}
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Convert to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"Failed to decode image from URL: {image_url}")
                return None
                
            return image
        except Exception as e:
            logger.error(f"Failed to load image from URL {image_url}: {e}")
            return None
    
    def _load_image_from_base64(self, base64_string: str) -> Optional[np.ndarray]:
        """Load image from base64 string and return as cv2 image (BGR)."""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode base64 image")
                return None
                
            return image
        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            return None
    
    def is_section_header(self, line: str) -> bool:
        """Check if a line is a section header."""
        line = line.strip().upper()
        
        # Common section headers
        section_keywords = [
            'APPETIZER', 'APPETIZERS', 'STARTER', 'STARTERS',
            'MAIN COURSE', 'MAIN COURSES', 'ENTREE', 'ENTREES',
            'DESSERT', 'DESSERTS', 'SWEET', 'SWEETS',
            'DRINK', 'DRINKS', 'BEVERAGE', 'BEVERAGES',
            'SOUP', 'SOUPS', 'SALAD', 'SALADS',
            'PIZZA', 'PIZZAS', 'PASTA', 'PASTAS',
            'SPECIAL', 'SPECIALS', 'COMBO', 'COMBOS'
        ]
        
        # Check if line matches section keywords
        for keyword in section_keywords:
            if keyword in line:
                return True
        
        # Check if line is all caps and reasonably short (likely a header)
        if line.isupper() and 3 <= len(line) <= 20 and line.isalpha():
            return True
            
        return False
    
    def extract_price_from_line(self, line: str) -> Optional[float]:
        """Extract price from a line of text."""
        # Price patterns for various formats
        price_patterns = [
            r'\$(\d+\.?\d*)',      # $12.99, $0.00
            r'(\d+\.?\d*)\s*\$',   # 12.99 $
            r'(\d+\.?\d*)\s*/-',   # 299 /-
            r'Rs\.?\s*(\d+\.?\d*)', # Rs. 299, Rs 299
            r'₹\s*(\d+\.?\d*)',    # ₹299
            r'(\d+\.?\d*)\s*(?:USD|INR|EUR|GBP)', # 299 USD
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    # Skip obviously wrong prices like 0.00
                    if price > 0:
                        return price
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def clean_item_name(self, name: str) -> str:
        """Clean menu item name by removing prices and artifacts."""
        if not name:
            return ""
        
        # Remove price patterns from name
        price_patterns = [
            r'\$\d+\.?\d*',
            r'\d+\.?\d*\s*\$',
            r'\d+\.?\d*\s*/-',
            r'Rs\.?\s*\d+\.?\d*',
            r'₹\s*\d+\.?\d*',
        ]
        
        cleaned = name
        for pattern in price_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-\.\&\']', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove generic placeholder text
        if 'food title' in cleaned.lower():
            return ""
        
        return cleaned
    
    def parse_menu_text(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse menu text into structured data."""
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        items = []
        current_section = "Menu"
        
        logger.debug(f"Parsing {len(lines)} lines of text")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty or very short lines
            if len(line) < 2:
                i += 1
                continue
            
            # Skip obvious non-menu content
            if any(skip in line.lower() for skip in ['deliver', 'order online', 'website', 'phone', 'address']):
                i += 1
                continue
            
            # Check if this line is a section header
            if self.is_section_header(line):
                current_section = line.title()
                logger.debug(f"Found section: {current_section}")
                i += 1
                continue
            
            # Try to parse as menu item
            price = self.extract_price_from_line(line)
            name = self.clean_item_name(line)
            
            # If no price in current line, check next line
            if not price and i + 1 < len(lines):
                next_line = lines[i + 1]
                price = self.extract_price_from_line(next_line)
                
                # If price is in next line, current line might be just the name
                if price:
                    name = self.clean_item_name(line)
                    i += 1  # Skip the price line
            
            # Only add if we have a valid name and price
            if name and price and price > 0:
                item = {
                    'name': name,
                    'price': price,
                    'currency': 'USD',
                    'availability': True,
                    'description': '',
                    'section': current_section
                }
                items.append(item)
                logger.debug(f"Added item: {name} - ${price} (Section: {current_section})")
            
            i += 1
        
        logger.debug(f"Total items parsed: {len(items)}")
        return items
    
    def _process_image_sync(self, image_url: str = None, image_base64: str = None, 
                           language: str = None, preprocessing: bool = False) -> Dict[str, Any]:
        """Process image using your exact original OCR approach."""
        start_time = time.time()
        
        try:
            # Load image
            if image_url:
                img = self._load_image_from_url(image_url)
            elif image_base64:
                img = self._load_image_from_base64(image_base64)
            else:
                raise ValueError("Either image_url or image_base64 must be provided")
            
            if img is None:
                raise ValueError("Failed to load image")
            
            # Your exact original OCR process
            # Convert BGR to RGB (exactly like your code)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use exact same tesseract config as your working code
            config = f'--oem 1 --psm 6'
            lang = language or self.language
            
            # Extract text exactly like your original code
            text = pytesseract.image_to_string(img_rgb, lang=lang, config=config)
            
            logger.debug(f"RAW OCR TEXT:\n{text}")
            
            # Parse using improved parsing function
            menu_items = self.parse_menu_text(text)
            
            logger.debug(f"PARSED MENU ITEMS: {len(menu_items)} items")
            for item in menu_items:
                logger.debug(f"Item: {item['name']} - ${item['price']} (Section: {item.get('section', 'Menu')})")
            
            processing_time = time.time() - start_time
            
            # Group items by section
            sections = {}
            for item in menu_items:
                section_name = item.pop('section', 'Menu')
                if section_name not in sections:
                    sections[section_name] = []
                sections[section_name].append(item)
            
            # If no items were parsed but we have text, create a fallback
            if not menu_items and text.strip():
                logger.warning("No items parsed, creating fallback entry")
                sections["Menu"] = [{
                    'name': "Menu items detected but could not parse details",
                    'price': 0.0,
                    'currency': 'USD',
                    'availability': True,
                    'description': f"Raw text: {text[:100]}..."
                }]
            
            return {
                'extracted_text': text,
                'confidence': 75.0,  # Default confidence
                'sections': sections,
                'total_items': len(menu_items),
                'processing_time': processing_time,
                'warnings': [] if menu_items else ["Could not parse menu items from extracted text"]
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {
                'extracted_text': '',
                'confidence': 0,
                'sections': {},
                'total_items': 0,
                'processing_time': time.time() - start_time,
                'warnings': [f"OCR processing failed: {str(e)}"]
            }
    
    async def process_image(self, image_url: str = None, image_base64: str = None, 
                           language: str = None, preprocessing: bool = False) -> Dict[str, Any]:
        """Async wrapper for image processing."""
        loop = asyncio.get_event_loop()
        
        # Run the synchronous OCR processing in a thread pool
        result = await loop.run_in_executor(
            self.executor,
            self._process_image_sync,
            image_url,
            image_base64,
            language,
            preprocessing
        )
        
        return result
    
    async def batch_process_images(self, image_sources: List[Dict[str, str]], 
                                  language: str = None, preprocessing: bool = False) -> List[Dict[str, Any]]:
        """Process multiple images concurrently."""
        tasks = []
        
        for i, source in enumerate(image_sources):
            task = self.process_image(
                image_url=source.get('url'),
                image_base64=source.get('base64'),
                language=language,
                preprocessing=preprocessing
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and add source indices
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process image {i+1}: {result}")
                processed_results.append({
                    'source_index': i,
                    'extracted_text': '',
                    'confidence': 0,
                    'sections': {},
                    'total_items': 0,
                    'processing_time': 0,
                    'warnings': [f"Processing failed: {str(result)}"]
                })
            else:
                result['source_index'] = i
                processed_results.append(result)
        
        return processed_results

# Global OCR service instance
ocr_service = OCRService()