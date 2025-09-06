"""OCR processor for extracting text from menu images using Tesseract."""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import requests
import base64
import io
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from loguru import logger
import time

from ..config.settings import settings
from ..utils.helpers import clean_text, extract_price

class OCRProcessor:
    """Advanced OCR processor for menu images with preprocessing capabilities."""
    
    def __init__(self):
        """Initialize OCR processor with Tesseract configuration."""
        # Set Tesseract command path
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        
        # OCR configuration
        self.config = settings.OCR_CONFIG
        self.language = settings.OCR_LANGUAGE
        
    def _load_image_from_url(self, image_url: str) -> Optional[Image.Image]:
        """Load image from URL."""
        try:
            headers = {'User-Agent': settings.USER_AGENT}
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            return image
        except Exception as e:
            logger.error(f"Failed to load image from URL {image_url}: {e}")
            return None
    
    def _load_image_from_base64(self, base64_string: str) -> Optional[Image.Image]:
        """Load image from base64 string."""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = settings.IMAGE_MAX_SIZE
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply preprocessing techniques
            cv_image = self._enhance_image_quality(cv_image)
            
            # Convert back to PIL
            processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original image: {e}")
            return image
    
    def _enhance_image_quality(self, cv_image: np.ndarray) -> np.ndarray:
        """Apply various image enhancement techniques."""
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    def _extract_text_with_confidence(self, image: Image.Image, config: str = None) -> Dict[str, Any]:
        """Extract text with confidence scores."""
        try:
            config_str = config or self.config
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, 
                lang=self.language, 
                config=config_str,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            
            for i, text in enumerate(data['text']):
                if text.strip():
                    texts.append(text)
                    confidences.append(data['conf'][i])
            
            extracted_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'text': extracted_text,
                'confidence': avg_confidence,
                'word_count': len(texts),
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {'text': '', 'confidence': 0, 'word_count': 0, 'raw_data': None}
    
    def _parse_menu_text(self, text: str) -> Dict[str, Any]:
        """Parse extracted text to identify menu items and structure."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        menu_items = []
        current_section = "Menu"
        warnings = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip very short lines (likely noise)
            if len(line) < 3:
                i += 1
                continue
            
            # Check if line looks like a section header
            if self._is_section_header(line):
                current_section = clean_text(line)
                i += 1
                continue
            
            # Try to parse as menu item
            item = self._parse_menu_item_line(line, lines[i:i+3])  # Look ahead 3 lines
            if item:
                item['section'] = current_section
                menu_items.append(item)
            else:
                warnings.append(f"Could not parse line: {line}")
            
            i += 1
        
        # Group items by section
        sections = {}
        for item in menu_items:
            section_name = item.pop('section', 'Menu')
            if section_name not in sections:
                sections[section_name] = []
            sections[section_name].append(item)
        
        return {
            'sections': sections,
            'total_items': len(menu_items),
            'warnings': warnings
        }
    
    def _is_section_header(self, line: str) -> bool:
        """Determine if a line is likely a section header."""
        # Common section header patterns
        section_patterns = [
            r'^[A-Z\s]+$',  # ALL CAPS
            r'.*(?:APPETIZER|STARTER|MAIN|ENTREE|DESSERT|BEVERAGE|DRINK|SPECIAL).*',
            r'^[A-Z][a-z\s]+:?$',  # Title case ending with optional colon
        ]
        
        # Check length (headers are usually short)
        if len(line) > 50:
            return False
        
        # Check patterns
        for pattern in section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Check if line has no price
        if not extract_price(line):
            # Check if next few lines have prices (indicating this might be a header)
            return True
        
        return False
    
    def _parse_menu_item_line(self, line: str, context_lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a single line as a menu item."""
        # Look for price in current line
        price = extract_price(line)
        
        # If no price in current line, check next line
        if not price and len(context_lines) > 1:
            price = extract_price(context_lines[1])
        
        # Extract name (everything before price indicators)
        name_part = line
        for price_pattern in settings.PRICE_PATTERNS:
            name_part = re.sub(price_pattern, '', name_part).strip()
        
        # Clean up name
        name = clean_text(name_part)
        
        # Skip if name is too short or looks like noise
        if not name or len(name) < 3:
            return None
        
        # Look for description in subsequent lines
        description = ""
        if len(context_lines) > 1:
            potential_desc = context_lines[1]
            # If next line doesn't have a price and isn't too short, it might be description
            if not extract_price(potential_desc) and len(potential_desc) > 10:
                description = clean_text(potential_desc)
        
        return {
            'name': name,
            'description': description,
            'price': price,
            'currency': 'USD',  # Default, could be enhanced with currency detection
            'availability': True
        }
    
    def process_image(self, image_url: str = None, image_base64: str = None, 
                     language: str = None, preprocessing: bool = True) -> Dict[str, Any]:
        """Process an image and extract menu information."""
        start_time = time.time()
        
        try:
            # Load image
            if image_url:
                image = self._load_image_from_url(image_url)
            elif image_base64:
                image = self._load_image_from_base64(image_base64)
            else:
                raise ValueError("Either image_url or image_base64 must be provided")
            
            if not image:
                raise ValueError("Failed to load image")
            
            # Use custom language if provided
            original_language = self.language
            if language:
                self.language = language
            
            # Preprocess image if requested
            if preprocessing:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image
            
            # Extract text with confidence
            ocr_result = self._extract_text_with_confidence(processed_image)
            
            # Parse menu structure
            parsed_result = self._parse_menu_text(ocr_result['text'])
            
            processing_time = time.time() - start_time
            
            # Restore original language
            self.language = original_language
            
            return {
                'extracted_text': ocr_result['text'],
                'confidence': ocr_result['confidence'],
                'sections': parsed_result['sections'],
                'total_items': parsed_result['total_items'],
                'processing_time': processing_time,
                'warnings': parsed_result['warnings']
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
    
    def batch_process_images(self, image_sources: List[Dict[str, str]], 
                           language: str = None, preprocessing: bool = True) -> List[Dict[str, Any]]:
        """Process multiple images in batch."""
        results = []
        
        for i, source in enumerate(image_sources):
            logger.info(f"Processing image {i+1}/{len(image_sources)}")
            
            try:
                result = self.process_image(
                    image_url=source.get('url'),
                    image_base64=source.get('base64'),
                    language=language,
                    preprocessing=preprocessing
                )
                result['source_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                results.append({
                    'source_index': i,
                    'extracted_text': '',
                    'confidence': 0,
                    'sections': {},
                    'total_items': 0,
                    'processing_time': 0,
                    'warnings': [f"Processing failed: {str(e)}"]
                })
        
        return results
    
    def validate_text_quality(self, text: str, min_length: int = 10, 
                            min_word_count: int = 3) -> Dict[str, Any]:
        """Validate the quality of extracted text."""
        words = text.split()
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'text_length': len(text),
            'word_count': len(words),
            'average_word_length': np.mean([len(word) for word in words]) if words else 0
        }
        
        # Check minimum length
        if len(text) < min_length:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Text too short: {len(text)} < {min_length}")
        
        # Check minimum word count
        if len(words) < min_word_count:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Too few words: {len(words)} < {min_word_count}")
        
        # Check for excessive special characters (might indicate OCR noise)
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        if special_char_ratio > 0.3:
            validation_result['issues'].append(f"High special character ratio: {special_char_ratio:.2f}")
        
        # Check for reasonable character distribution
        if text:
            alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
            if alpha_ratio < 0.5:
                validation_result['issues'].append(f"Low alphabetic character ratio: {alpha_ratio:.2f}")
        
        return validation_result
