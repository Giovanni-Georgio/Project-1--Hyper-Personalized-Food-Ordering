"""Configuration settings for the menu scraper module."""

import os
from typing import List, Dict, Any
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

class Settings:
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/menu_db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Selenium Configuration
    SELENIUM_TIMEOUT = int(os.getenv("SELENIUM_TIMEOUT", "30"))
    HEADLESS_BROWSER = os.getenv("HEADLESS_BROWSER", "true").lower() == "true"
    BROWSER_TYPE = os.getenv("BROWSER_TYPE", "chrome")  # chrome, firefox, edge
    
    # OCR Configuration
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")  # Adjust path as needed
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
    OCR_CONFIG = "--psm 6"  # Page segmentation mode
    
    # Scraping Configuration
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))  # Seconds between requests
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # File Storage
    UPLOAD_DIR = DATA_DIR / "uploads"
    PROCESSED_DIR = DATA_DIR / "processed"
    CACHE_DIR = DATA_DIR / "cache"
    
    # Ensure storage directories exist
    UPLOAD_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "menu_scraper.log"
    
    # Supported Restaurant Platforms
    SUPPORTED_PLATFORMS: List[str] = [
        "zomato",
        "ubereats", 
        "grubhub",
        "doordash",
        "swiggy",
        "foodpanda",
        "deliveroo",
        "justeat",
        "seamless",
        "postmates"
    ]
    
    # Platform-specific selectors
    PLATFORM_SELECTORS: Dict[str, Dict[str, str]] = {
        "zomato": {
            "menu_container": ".menu-container",
            "menu_item": ".menu-item",
            "item_name": ".dish-name", 
            "item_price": ".dish-price",
            "item_description": ".dish-desc",
            "menu_section": ".menu-section-name"
        },
        "ubereats": {
            "menu_container": "[data-testid='menu-item']",
            "menu_item": "[data-testid='store-menu-item']",
            "item_name": "h3[data-testid='rich-text']",
            "item_price": "[data-testid='price-text']",
            "item_description": "[data-testid='menu-item-description']",
            "menu_section": "[data-testid='store-menu-category-title']"
        },
        "doordash": {
            "menu_container": "[data-anchor-id='MenuItem']",
            "menu_item": "[data-testid='menu-item']",
            "item_name": "[data-testid='menu-item-name']",
            "item_price": "[data-testid='menu-item-price']", 
            "item_description": "[data-testid='menu-item-description']",
            "menu_section": "[data-testid='menu-category-header']"
        }
    }
    
    # Data cleaning configuration
    PRICE_PATTERNS = [
        r'\$[\d,]+\.?\d*',  # $12.99, $1,299
        r'₹[\d,]+\.?\d*',   # ₹299
        r'€[\d,]+\.?\d*',   # €15.50
        r'£[\d,]+\.?\d*',   # £8.99
        r'[\d,]+\.?\d*\s*(?:USD|INR|EUR|GBP)',  # 299 INR
    ]
    
    # Image processing settings
    IMAGE_MAX_SIZE = (1920, 1080)
    IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'bmp']
    
    # Cache settings
    CACHE_EXPIRY_HOURS = 24
    
settings = Settings()
