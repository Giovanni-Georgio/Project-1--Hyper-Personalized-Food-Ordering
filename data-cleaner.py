"""Data cleaning and processing utilities for scraped menu data."""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import unicodedata
from datetime import datetime

from ..config.settings import settings

class DataCleaner:
    """Comprehensive data cleaning and processing for menu data."""
    
    def __init__(self):
        """Initialize data cleaner with configuration."""
        self.price_patterns = settings.PRICE_PATTERNS
        
    def clean_restaurant_data(self, restaurant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize restaurant information."""
        cleaned = restaurant_data.copy()
        
        # Clean restaurant name
        if 'name' in cleaned:
            cleaned['name'] = self.clean_text(cleaned['name'])
            cleaned['name'] = self.title_case_restaurant_name(cleaned['name'])
        
        # Standardize platform name
        if 'platform' in cleaned:
            cleaned['platform'] = cleaned['platform'].lower().strip()
        
        # Clean and validate URL
        if 'url' in cleaned:
            cleaned['url'] = self.clean_url(cleaned['url'])
        
        # Clean cuisine type
        if 'cuisine_type' in cleaned:
            cleaned['cuisine_type'] = self.clean_cuisine_type(cleaned['cuisine_type'])
        
        # Validate and clean rating
        if 'rating' in cleaned and cleaned['rating']:
            cleaned['rating'] = self.validate_rating(cleaned['rating'])
        
        # Clean address
        if 'address' in cleaned:
            cleaned['address'] = self.clean_address(cleaned['address'])
        
        # Clean and validate phone
        if 'phone' in cleaned:
            cleaned['phone'] = self.clean_phone_number(cleaned['phone'])
        
        # Clean delivery time
        if 'delivery_time' in cleaned:
            cleaned['delivery_time'] = self.clean_delivery_time(cleaned['delivery_time'])
        
        # Validate monetary values
        for field in ['minimum_order', 'delivery_fee']:
            if field in cleaned and cleaned[field]:
                cleaned[field] = self.validate_price(cleaned[field])
        
        return cleaned
    
    def clean_menu_section_data(self, section_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean menu section information."""
        cleaned = section_data.copy()
        
        # Clean section name
        if 'name' in cleaned:
            cleaned['name'] = self.clean_text(cleaned['name'])
            cleaned['name'] = self.title_case_section_name(cleaned['name'])
        
        # Clean description
        if 'description' in cleaned:
            cleaned['description'] = self.clean_text(cleaned['description'])
        
        # Validate display order
        if 'display_order' in cleaned:
            cleaned['display_order'] = max(0, int(cleaned.get('display_order', 0)))
        
        return cleaned
    
    def clean_menu_item_data(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean individual menu item data."""
        cleaned = item_data.copy()
        
        # Clean item name
        if 'name' in cleaned:
            cleaned['name'] = self.clean_menu_item_name(cleaned['name'])
        
        # Clean description
        if 'description' in cleaned:
            cleaned['description'] = self.clean_text(cleaned['description'])
        
        # Clean and validate price
        if 'price' in cleaned:
            cleaned['price'] = self.validate_price(cleaned['price'])
        
        if 'original_price' in cleaned:
            cleaned['original_price'] = self.validate_price(cleaned['original_price'])
        
        # Standardize currency
        if 'currency' in cleaned:
            cleaned['currency'] = self.standardize_currency(cleaned['currency'])
        
        # Clean image URL
        if 'image_url' in cleaned:
            cleaned['image_url'] = self.clean_url(cleaned['image_url'])
        
        # Clean preparation time
        if 'preparation_time' in cleaned:
            cleaned['preparation_time'] = self.clean_preparation_time(cleaned['preparation_time'])
        
        # Validate calories
        if 'calories' in cleaned and cleaned['calories']:
            try:
                cleaned['calories'] = max(0, int(cleaned['calories']))
            except (ValueError, TypeError):
                cleaned['calories'] = None
        
        # Clean allergens
        if 'allergens' in cleaned:
            cleaned['allergens'] = self.clean_allergens_list(cleaned['allergens'])
        
        # Detect dietary preferences
        cleaned.update(self.detect_dietary_preferences(cleaned))
        
        return cleaned
    
    def clean_text(self, text: str) -> str:
        """General text cleaning function."""
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Remove common OCR artifacts
        text = re.sub(r'[|\\/_]+', ' ', text)
        text = re.sub(r'[^\w\s\-.,()&$€£¥₹]', '', text)
        
        return text.strip()
    
    def clean_menu_item_name(self, name: str) -> str:
        """Specifically clean menu item names."""
        if not name:
            return ""
        
        # Basic cleaning
        name = self.clean_text(name)
        
        # Remove price information from name
        for pattern in self.price_patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Remove common suffixes that might be artifacts
        suffixes_to_remove = [
            r'\s*\([^)]*\)$',  # Remove trailing parentheses
            r'\s*-\s*$',       # Remove trailing dashes
            r'\s*\.\s*$',      # Remove trailing periods
        ]
        
        for suffix in suffixes_to_remove:
            name = re.sub(suffix, '', name)
        
        # Title case for better presentation
        name = self.smart_title_case(name)
        
        return name.strip()
    
    def smart_title_case(self, text: str) -> str:
        """Apply smart title case preserving certain words."""
        if not text:
            return ""
        
        # Words that should remain lowercase (unless at start)
        lowercase_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'of', 'to', 'with'}
        
        words = text.lower().split()
        result = []
        
        for i, word in enumerate(words):
            if i == 0 or word not in lowercase_words:
                result.append(word.capitalize())
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def title_case_restaurant_name(self, name: str) -> str:
        """Title case for restaurant names with special handling."""
        if not name:
            return ""
        
        # Handle common restaurant name patterns
        name = self.smart_title_case(name)
        
        # Handle common abbreviations
        abbreviations = {
            r'\\bMc([a-z])': r'Mc\\1',  # McDonald's style
            r'\\bCafe\\b': 'Café',
            r'\\bBbq\\b': 'BBQ',
            r'\\bKfc\\b': 'KFC',
        }
        
        for pattern, replacement in abbreviations.items():
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        
        return name
    
    def title_case_section_name(self, name: str) -> str:
        """Title case for menu section names."""
        if not name:
            return ""
        
        # Remove colons and other punctuation
        name = re.sub(r'[:.]+$', '', name)
        
        return self.smart_title_case(name)
    
    def validate_price(self, price: Union[str, float, int]) -> Optional[float]:
        """Extract and validate price from various formats."""
        if price is None:
            return None
        
        if isinstance(price, (int, float)):
            return float(price) if price >= 0 else None
        
        if isinstance(price, str):
            # Extract numeric price using patterns
            for pattern in self.price_patterns:
                match = re.search(pattern, price)
                if match:
                    # Extract just the numeric part
                    numeric_part = re.search(r'[\d,]+\.?\d*', match.group())
                    if numeric_part:
                        try:
                            price_value = float(numeric_part.group().replace(',', ''))
                            return price_value if price_value >= 0 else None
                        except ValueError:
                            continue
        
        return None
    
    def standardize_currency(self, currency: str) -> str:
        """Standardize currency codes."""
        if not currency:
            return "USD"
        
        currency_mapping = {
            '$': 'USD',
            '₹': 'INR', 
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            'dollar': 'USD',
            'rupee': 'INR',
            'euro': 'EUR',
            'pound': 'GBP',
        }
        
        currency = currency.lower().strip()
        return currency_mapping.get(currency, currency.upper())
    
    def validate_rating(self, rating: Union[str, float, int]) -> Optional[float]:
        """Validate and clean rating values."""
        if rating is None:
            return None
        
        try:
            if isinstance(rating, str):
                # Extract numeric rating from string
                match = re.search(r'(\d+\.?\d*)', rating)
                if match:
                    rating = float(match.group(1))
                else:
                    return None
            
            rating = float(rating)
            
            # Validate range (assume 0-5 scale, but handle 0-10 scale too)
            if 0 <= rating <= 5:
                return rating
            elif 5 < rating <= 10:
                return rating / 2  # Convert 10-scale to 5-scale
            else:
                return None
                
        except (ValueError, TypeError):
            return None
    
    def clean_phone_number(self, phone: str) -> Optional[str]:
        """Clean and validate phone numbers."""
        if not phone:
            return None
        
        # Remove all non-digit characters except + for country code
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Basic validation (must have at least 7 digits)
        if len(re.sub(r'[^\d]', '', cleaned)) < 7:
            return None
        
        return cleaned
    
    def clean_address(self, address: str) -> Optional[str]:
        """Clean address information."""
        if not address:
            return None
        
        # Basic text cleaning
        address = self.clean_text(address)
        
        # Remove redundant spacing around commas
        address = re.sub(r'\s*,\s*', ', ', address)
        
        return address.strip() if address else None
    
    def clean_url(self, url: str) -> Optional[str]:
        """Clean and validate URLs."""
        if not url:
            return None
        
        url = url.strip()
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Basic URL validation
        url_pattern = r'^https?:\/\/.+\..+'
        if re.match(url_pattern, url):
            return url
        
        return None
    
    def clean_cuisine_type(self, cuisine: str) -> Optional[str]:
        """Clean and standardize cuisine types."""
        if not cuisine:
            return None
        
        cuisine = self.clean_text(cuisine)
        cuisine = self.smart_title_case(cuisine)
        
        # Standardize common cuisine names
        cuisine_mapping = {
            'chinese': 'Chinese',
            'italian': 'Italian', 
            'mexican': 'Mexican',
            'indian': 'Indian',
            'american': 'American',
            'fast food': 'Fast Food',
            'pizza': 'Pizza',
            'burger': 'Burgers',
            'seafood': 'Seafood',
        }
        
        lower_cuisine = cuisine.lower()
        return cuisine_mapping.get(lower_cuisine, cuisine)
    
    def clean_delivery_time(self, delivery_time: str) -> Optional[str]:
        """Clean delivery time information."""
        if not delivery_time:
            return None
        
        delivery_time = self.clean_text(delivery_time)
        
        # Standardize format (e.g., "30-45 min", "45 mins")
        time_pattern = r'(\d+)[-–]?(\d+)?\s*(min|minute|minutes)'
        match = re.search(time_pattern, delivery_time, re.IGNORECASE)
        
        if match:
            min_time = match.group(1)
            max_time = match.group(2) or min_time
            return f"{min_time}-{max_time} min"
        
        return delivery_time
    
    def clean_preparation_time(self, prep_time: str) -> Optional[str]:
        """Clean preparation time information."""
        return self.clean_delivery_time(prep_time)  # Same logic
    
    def clean_allergens_list(self, allergens: Union[str, List[str]]) -> List[str]:
        """Clean and standardize allergen information."""
        if not allergens:
            return []
        
        if isinstance(allergens, str):
            # Split by common delimiters
            allergens = re.split(r'[,;]', allergens)
        
        cleaned_allergens = []
        for allergen in allergens:
            if isinstance(allergen, str):
                allergen = self.clean_text(allergen).strip()
                if allergen:
                    cleaned_allergens.append(allergen.title())
        
        return list(set(cleaned_allergens))  # Remove duplicates
    
    def detect_dietary_preferences(self, item_data: Dict[str, Any]) -> Dict[str, bool]:
        """Detect dietary preferences from item name and description."""
        text_to_analyze = ""
        
        if item_data.get('name'):
            text_to_analyze += item_data['name'].lower() + " "
        
        if item_data.get('description'):
            text_to_analyze += item_data['description'].lower() + " "
        
        dietary_indicators = {
            'is_vegetarian': [
                'veg', 'vegetarian', 'veggie', 'no meat', 'plant based',
                'meatless', 'garden', 'greens'
            ],
            'is_vegan': [
                'vegan', 'plant based', 'dairy free', 'no dairy',
                'no cheese', 'no milk', 'no butter'
            ],
            'is_spicy': [
                'spicy', 'hot', 'chili', 'jalapeño', 'pepper', 'sriracha',
                'wasabi', 'fire', 'blazing', 'ghost pepper'
            ]
        }
        
        result = {}
        for preference, indicators in dietary_indicators.items():
            result[preference] = any(indicator in text_to_analyze for indicator in indicators)
        
        return result
    
    def process_batch_data(self, data_list: List[Dict[str, Any]], 
                          data_type: str) -> List[Dict[str, Any]]:
        """Process a batch of data items."""
        cleaned_data = []
        
        for item in data_list:
            try:
                if data_type == 'restaurant':
                    cleaned_item = self.clean_restaurant_data(item)
                elif data_type == 'section':
                    cleaned_item = self.clean_menu_section_data(item)
                elif data_type == 'item':
                    cleaned_item = self.clean_menu_item_data(item)
                else:
                    cleaned_item = item
                
                cleaned_data.append(cleaned_item)
                
            except Exception as e:
                logger.warning(f"Failed to clean {data_type} data: {e}")
                cleaned_data.append(item)  # Keep original if cleaning fails
        
        return cleaned_data
    
    def validate_completeness(self, data: Dict[str, Any], 
                            required_fields: List[str]) -> Dict[str, Any]:
        """Validate data completeness."""
        validation_result = {
            'is_complete': True,
            'missing_fields': [],
            'empty_fields': [],
            'completeness_score': 0.0
        }
        
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif not data[field]:
                empty_fields.append(field)
        
        validation_result['missing_fields'] = missing_fields
        validation_result['empty_fields'] = empty_fields
        validation_result['is_complete'] = len(missing_fields) == 0 and len(empty_fields) == 0
        
        # Calculate completeness score
        total_fields = len(required_fields)
        complete_fields = total_fields - len(missing_fields) - len(empty_fields)
        validation_result['completeness_score'] = complete_fields / total_fields if total_fields > 0 else 0.0
        
        return validation_result