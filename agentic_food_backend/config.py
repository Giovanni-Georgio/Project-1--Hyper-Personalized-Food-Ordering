import os
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:test123@localhost/agentic_food_ordering")
SYNC_DATABASE_URL = os.getenv("SYNC_DATABASE_URL", "postgresql://postgres:test123@localhost/agentic_food_ordering")

SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, UPLOAD_DIR, PROCESSED_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# OCR Configuration
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")  # Adjust for Windows: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
OCR_CONFIG = "--psm 6"  # Page segmentation mode

# Selenium Configuration
SELENIUM_TIMEOUT = int(os.getenv("SELENIUM_TIMEOUT", "30"))
HEADLESS_BROWSER = os.getenv("HEADLESS_BROWSER", "true").lower() == "true"

# Scraping Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Image Processing Settings
IMAGE_MAX_SIZE = (1920, 1080)
IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'bmp']

# Price extraction patterns
PRICE_PATTERNS = [
    r'\$[\d,]+\.?\d*',  # $12.99, $1,299
    r'₹[\d,]+\.?\d*',   # ₹299
    r'€[\d,]+\.?\d*',   # €15.50
    r'£[\d,]+\.?\d*',   # £8.99
    r'[\d,]+\.?\d*\s*(?:USD|INR|EUR|GBP)',  # 299 INR
]

# Supported Restaurant Platforms
SUPPORTED_PLATFORMS = [
    "zomato", "ubereats", "grubhub", "doordash", 
    "swiggy", "foodpanda", "deliveroo", "justeat"
]

# Platform-specific selectors
PLATFORM_SELECTORS = {
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
    }
}