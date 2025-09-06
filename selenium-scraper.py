"""Selenium-based web scraper for restaurant menus."""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import re
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import settings
from ..data.models import RestaurantCreate, MenuSectionCreate, MenuItemCreate
from ..utils.helpers import detect_platform, clean_text, extract_price
from .base_scraper import BaseScraper

class SeleniumScraper(BaseScraper):
    """Advanced Selenium scraper for dynamic restaurant menu content."""
    
    def __init__(self):
        """Initialize the Selenium scraper."""
        self.driver = None
        self.wait = None
        
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome WebDriver with optimal configuration."""
        chrome_options = Options()
        
        # Performance optimizations
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        # Speed optimizations
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")  # Can be removed if JS is needed
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.media_stream": 2,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Headless mode
        if settings.HEADLESS_BROWSER:
            chrome_options.add_argument("--headless")
            
        # User agent
        chrome_options.add_argument(f"--user-agent={settings.USER_AGENT}")
        
        # Window size
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Setup service
        service = Service(ChromeDriverManager().install())
        
        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(settings.SELENIUM_TIMEOUT)
            driver.implicitly_wait(10)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            raise
    
    def _get_platform_selectors(self, platform: str) -> Dict[str, str]:
        """Get CSS selectors for the specified platform."""
        return settings.PLATFORM_SELECTORS.get(platform, {})
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _load_page(self, url: str) -> None:
        """Load a webpage with retry logic."""
        try:
            logger.info(f"Loading URL: {url}")
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, settings.SELENIUM_TIMEOUT).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Additional wait for dynamic content
            time.sleep(3)
            
        except TimeoutException:
            logger.warning(f"Page load timeout for: {url}")
            raise
        except WebDriverException as e:
            logger.error(f"WebDriver error loading {url}: {e}")
            raise
    
    def _handle_popups_and_cookies(self) -> None:
        """Handle common popups, cookie banners, and overlays."""
        popup_selectors = [
            # Common cookie banner selectors
            "button[id*='cookie']",
            "button[class*='cookie']", 
            "button[id*='accept']",
            "button[class*='accept']",
            ".cookie-banner button",
            "#cookie-consent button",
            
            # Common popup selectors
            ".modal button",
            ".popup button",
            ".overlay button",
            "[data-testid*='close']",
            "[aria-label*='close' i]",
            "[aria-label*='dismiss' i]",
            "button[class*='close']",
            ".close-button",
            
            # Platform-specific
            "[data-testid='close-button']",
            "[data-testid='dismiss-button']", 
            "button[aria-label='Close']"
        ]
        
        for selector in popup_selectors:
            try:
                element = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                element.click()
                logger.info(f"Closed popup using selector: {selector}")
                time.sleep(1)
                break
            except (TimeoutException, NoSuchElementException):
                continue
    
    def _wait_for_menu_content(self, platform: str) -> None:
        """Wait for menu content to load based on platform."""
        selectors = self._get_platform_selectors(platform)
        
        if not selectors:
            # Generic wait for menu content
            menu_indicators = [
                "[class*='menu']",
                "[data-testid*='menu']",
                "[id*='menu']",
                ".dish",
                ".item",
                ".food"
            ]
        else:
            menu_indicators = [
                selectors.get("menu_container", ""),
                selectors.get("menu_item", ""),
                selectors.get("menu_section", "")
            ]
        
        for selector in menu_indicators:
            if selector:
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"Menu content loaded, found: {selector}")
                    return
                except TimeoutException:
                    continue
        
        logger.warning("No menu content indicators found, proceeding anyway")
    
    def _extract_restaurant_info(self, url: str, platform: str) -> RestaurantCreate:
        """Extract basic restaurant information."""
        restaurant_info = {
            "name": "Unknown Restaurant",
            "platform": platform,
            "url": url,
            "cuisine_type": None,
            "rating": None,
            "address": None,
            "phone": None,
            "delivery_time": None,
            "minimum_order": None,
            "delivery_fee": None
        }
        
        # Platform-specific extraction logic
        if platform == "zomato":
            restaurant_info.update(self._extract_zomato_restaurant_info())
        elif platform == "ubereats":
            restaurant_info.update(self._extract_ubereats_restaurant_info())
        elif platform == "doordash":
            restaurant_info.update(self._extract_doordash_restaurant_info())
        else:
            restaurant_info.update(self._extract_generic_restaurant_info())
        
        return RestaurantCreate(**restaurant_info)
    
    def _extract_generic_restaurant_info(self) -> Dict[str, Any]:
        """Extract restaurant info using generic selectors."""
        info = {}
        
        # Try to find restaurant name
        name_selectors = [
            "h1", "h2", "[class*='restaurant-name']", "[class*='store-name']",
            "[data-testid*='restaurant-name']", "[data-testid*='store-name']",
            "title", ".title", "#restaurant-name", ".restaurant-title"
        ]
        
        for selector in name_selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                if element.text.strip():
                    info["name"] = clean_text(element.text)
                    break
            except NoSuchElementException:
                continue
        
        # Try to find rating
        rating_selectors = [
            "[class*='rating']", "[data-testid*='rating']", 
            ".star", ".stars", "[class*='star']"
        ]
        
        for selector in rating_selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                rating_text = element.text or element.get_attribute("title") or element.get_attribute("aria-label")
                if rating_text:
                    rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                    if rating_match:
                        info["rating"] = float(rating_match.group(1))
                        break
            except (NoSuchElementException, ValueError):
                continue
        
        return info
    
    def _extract_zomato_restaurant_info(self) -> Dict[str, Any]:
        """Extract Zomato-specific restaurant information."""
        info = {}
        
        try:
            # Restaurant name
            name_element = self.driver.find_element(By.CSS_SELECTOR, "h1")
            info["name"] = clean_text(name_element.text)
        except NoSuchElementException:
            pass
        
        try:
            # Rating
            rating_element = self.driver.find_element(By.CSS_SELECTOR, "[class*='rating']")
            rating_text = rating_element.text
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                info["rating"] = float(rating_match.group(1))
        except (NoSuchElementException, ValueError):
            pass
        
        try:
            # Cuisine type
            cuisine_element = self.driver.find_element(By.CSS_SELECTOR, "[class*='cuisine']")
            info["cuisine_type"] = clean_text(cuisine_element.text)
        except NoSuchElementException:
            pass
        
        return info
    
    def _extract_ubereats_restaurant_info(self) -> Dict[str, Any]:
        """Extract UberEats-specific restaurant information."""
        info = {}
        
        try:
            name_element = self.driver.find_element(By.CSS_SELECTOR, "h1[data-testid='store-title']")
            info["name"] = clean_text(name_element.text)
        except NoSuchElementException:
            pass
        
        try:
            rating_element = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='rating']")
            rating_text = rating_element.text
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                info["rating"] = float(rating_match.group(1))
        except (NoSuchElementException, ValueError):
            pass
        
        return info
    
    def _extract_doordash_restaurant_info(self) -> Dict[str, Any]:
        """Extract DoorDash-specific restaurant information."""
        info = {}
        
        try:
            name_element = self.driver.find_element(By.CSS_SELECTOR, "h1[data-testid='store-name']")
            info["name"] = clean_text(name_element.text)
        except NoSuchElementException:
            pass
        
        return info
    
    def _extract_menu_sections(self, platform: str) -> List[Dict[str, Any]]:
        """Extract menu sections and items."""
        selectors = self._get_platform_selectors(platform)
        sections = []
        
        if selectors:
            sections = self._extract_with_selectors(selectors)
        else:
            sections = self._extract_generic_menu()
        
        return sections
    
    def _extract_with_selectors(self, selectors: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract menu using platform-specific selectors."""
        sections = []
        
        try:
            # Find menu sections
            section_elements = self.driver.find_elements(By.CSS_SELECTOR, selectors.get("menu_section", ""))
            
            if not section_elements:
                # If no sections found, create a default section
                sections.append(self._extract_section_items("Menu", selectors))
            else:
                for idx, section_element in enumerate(section_elements):
                    section_name = clean_text(section_element.text) or f"Section {idx + 1}"
                    section_data = self._extract_section_items(section_name, selectors, section_element)
                    if section_data["items"]:
                        sections.append(section_data)
        
        except Exception as e:
            logger.error(f"Error extracting with selectors: {e}")
            # Fallback to generic extraction
            sections = self._extract_generic_menu()
        
        return sections
    
    def _extract_section_items(self, section_name: str, selectors: Dict[str, str], section_element=None) -> Dict[str, Any]:
        """Extract items for a specific section."""
        section_data = {
            "name": section_name,
            "description": "",
            "display_order": 0,
            "items": []
        }
        
        # Find menu items within section or globally
        search_context = section_element if section_element else self.driver
        item_elements = search_context.find_elements(By.CSS_SELECTOR, selectors.get("menu_item", ""))
        
        for item_element in item_elements:
            try:
                item_data = self._extract_item_details(item_element, selectors)
                if item_data and item_data.get("name"):
                    section_data["items"].append(item_data)
            except Exception as e:
                logger.warning(f"Error extracting item: {e}")
                continue
        
        return section_data
    
    def _extract_item_details(self, item_element, selectors: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract details for a single menu item."""
        try:
            # Extract item name
            name_element = item_element.find_element(By.CSS_SELECTOR, selectors.get("item_name", ""))
            name = clean_text(name_element.text)
            if not name:
                return None
            
            # Extract price
            price = None
            try:
                price_element = item_element.find_element(By.CSS_SELECTOR, selectors.get("item_price", ""))
                price_text = price_element.text
                price = extract_price(price_text)
            except NoSuchElementException:
                pass
            
            # Extract description
            description = ""
            try:
                desc_element = item_element.find_element(By.CSS_SELECTOR, selectors.get("item_description", ""))
                description = clean_text(desc_element.text)
            except NoSuchElementException:
                pass
            
            # Extract image URL
            image_url = None
            try:
                img_element = item_element.find_element(By.TAG_NAME, "img")
                image_url = img_element.get_attribute("src") or img_element.get_attribute("data-src")
            except NoSuchElementException:
                pass
            
            return {
                "name": name,
                "description": description,
                "price": price,
                "currency": "USD",  # Default, can be improved with detection
                "image_url": image_url,
                "availability": True,
                "display_order": 0
            }
            
        except Exception as e:
            logger.warning(f"Error extracting item details: {e}")
            return None
    
    def _extract_generic_menu(self) -> List[Dict[str, Any]]:
        """Generic menu extraction when platform-specific selectors fail."""
        sections = []
        
        # Generic selectors for menu items
        generic_selectors = [
            "[class*='menu-item']", "[class*='dish']", "[class*='food-item']",
            "[data-testid*='menu-item']", "[data-testid*='dish']",
            ".item", ".product", "[class*='product']"
        ]
        
        items_found = []
        for selector in generic_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    for element in elements:
                        item = self._extract_generic_item(element)
                        if item and item.get("name"):
                            items_found.append(item)
                    break
            except Exception:
                continue
        
        if items_found:
            sections.append({
                "name": "Menu",
                "description": "",
                "display_order": 0,
                "items": items_found
            })
        
        return sections
    
    def _extract_generic_item(self, element) -> Optional[Dict[str, Any]]:
        """Extract item using generic selectors."""
        try:
            # Try to find text that looks like a menu item
            text_content = element.text.strip()
            if not text_content or len(text_content) < 3:
                return None
            
            # Split by lines to separate name, description, price
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            
            if not lines:
                return None
            
            name = lines[0]
            description = ""
            price = None
            
            # Look for price in the text
            for line in lines:
                extracted_price = extract_price(line)
                if extracted_price:
                    price = extracted_price
                    break
            
            # Description is usually the longest non-price line
            if len(lines) > 1:
                for line in lines[1:]:
                    if not extract_price(line) and len(line) > len(description):
                        description = line
            
            return {
                "name": clean_text(name),
                "description": clean_text(description),
                "price": price,
                "currency": "USD",
                "availability": True,
                "display_order": 0
            }
            
        except Exception:
            return None
    
    async def scrape_menu(self, url: str) -> Dict[str, Any]:
        """Main method to scrape menu from a restaurant URL."""
        start_time = time.time()
        
        try:
            # Setup driver
            self.driver = self._setup_driver()
            self.wait = WebDriverWait(self.driver, settings.SELENIUM_TIMEOUT)
            
            # Detect platform
            platform = detect_platform(url)
            logger.info(f"Detected platform: {platform} for URL: {url}")
            
            # Load page
            self._load_page(url)
            
            # Handle popups and cookies
            self._handle_popups_and_cookies()
            
            # Wait for menu content
            self._wait_for_menu_content(platform)
            
            # Extract restaurant info
            restaurant_info = self._extract_restaurant_info(url, platform)
            
            # Extract menu sections
            menu_sections = self._extract_menu_sections(platform)
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                "restaurant": restaurant_info.dict(),
                "sections": menu_sections,
                "total_items": sum(len(section.get("items", [])) for section in menu_sections),
                "total_sections": len(menu_sections),
                "processing_time": processing_time,
                "platform": platform,
                "warnings": []
            }
            
            logger.info(f"Scraping completed: {result['total_items']} items in {result['total_sections']} sections")
            return result
            
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            raise
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except Exception:
                    pass
    
    def __del__(self):
        """Cleanup driver on object destruction."""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass