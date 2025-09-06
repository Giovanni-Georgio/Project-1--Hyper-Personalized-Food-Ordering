"""
Test module for Menu Data Scraping functionality
Run tests with: pytest test_menu_scraper.py -v
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Import modules to test
from menu_scraper import MenuScraper, OCRProcessor, MenuDataStructure, DataCleaner
from database_storage import MenuDatabase, Restaurant, MenuItem
from utils import validate_url, clean_price, categorize_menu_item, extract_dietary_info


class TestMenuDataStructure:
    """Test menu data structure creation"""

    def test_create_menu_item(self):
        """Test menu item creation"""
        item = MenuDataStructure.create_menu_item(
            name="Test Pizza",
            price=12.99,
            description="Delicious pizza",
            category="Pizza"
        )

        assert item["name"] == "Test Pizza"
        assert item["price"] == 12.99
        assert item["description"] == "Delicious pizza"
        assert item["category"] == "Pizza"
        assert "scraped_at" in item

    def test_create_restaurant_menu(self):
        """Test restaurant menu creation"""
        menu_items = [
            MenuDataStructure.create_menu_item("Item 1", 10.0),
            MenuDataStructure.create_menu_item("Item 2", 15.0)
        ]

        restaurant = MenuDataStructure.create_restaurant_menu(
            restaurant_name="Test Restaurant",
            location="Test Location",
            menu_items=menu_items
        )

        assert restaurant["restaurant_name"] == "Test Restaurant"
        assert restaurant["location"] == "Test Location"
        assert restaurant["total_items"] == 2
        assert len(restaurant["menu_items"]) == 2


class TestOCRProcessor:
    """Test OCR processing functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.ocr = OCRProcessor()

    def test_preprocess_image(self):
        """Test image preprocessing"""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = self.ocr.preprocess_image(test_image)

        assert processed.shape == (300, 300)  # Should be resized
        assert len(processed.shape) == 2  # Should be grayscale

    def test_parse_menu_from_text(self):
        """Test menu parsing from OCR text"""
        test_text = """
        Pizza Margherita $12.99
        Caesar Salad $8.50
        Chicken Wings $14.00
        """

        items = self.ocr.parse_menu_from_text(test_text)

        assert len(items) == 3
        assert items[0]["name"] == "Pizza Margherita"
        assert items[0]["price"] == 12.99
        assert items[1]["price"] == 8.50


class TestDataCleaner:
    """Test data cleaning functionality"""

    def test_clean_menu_item(self):
        """Test menu item cleaning"""
        dirty_item = {
            "name": "  Pizza @ Margherita!!! ",
            "description": "  Fresh & delicious  ",
            "price": "12.99"
        }

        clean_item = DataCleaner.clean_menu_item(dirty_item)

        assert clean_item["name"] == "Pizza  Margherita"
        assert clean_item["description"] == "Fresh  delicious"
        assert clean_item["price"] == 12.99

    def test_remove_duplicates(self):
        """Test duplicate removal"""
        menu_data = {
            "restaurant_name": "Test",
            "menu_items": [
                {"name": "Pizza", "price": 12.99},
                {"name": "Pizza", "price": 12.99},  # Duplicate
                {"name": "Salad", "price": 8.99}
            ]
        }

        cleaned = DataCleaner.remove_duplicates(menu_data)

        assert len(cleaned["menu_items"]) == 2
        assert cleaned["total_items"] == 2


class TestDatabase:
    """Test database functionality"""

    def setup_method(self):
        """Setup test database"""
        self.db = MenuDatabase("sqlite:///:memory:")  # In-memory database

    def test_store_and_retrieve_restaurant(self):
        """Test storing and retrieving restaurant data"""
        menu_data = {
            "restaurant_name": "Test Restaurant",
            "location": "Test Location",
            "cuisine_type": "Italian",
            "menu_items": [
                {"name": "Pizza", "price": 12.99, "category": "Main"},
                {"name": "Salad", "price": 8.99, "category": "Appetizer"}
            ],
            "metadata": {"scraping_source": "test", "scraping_method": "manual"}
        }

        # Store restaurant
        restaurant_id = self.db.store_restaurant_menu(menu_data)
        assert restaurant_id is not None

        # Retrieve restaurant
        retrieved = self.db.get_restaurant_by_id(restaurant_id)
        assert retrieved is not None
        assert retrieved["name"] == "Test Restaurant"
        assert len(retrieved["menu_items"]) == 2

    def test_search_functionality(self):
        """Test search functionality"""
        # Store test data first
        menu_data = {
            "restaurant_name": "Pizza Place",
            "menu_items": [
                {"name": "Margherita Pizza", "price": 12.99, "description": "Classic pizza"},
                {"name": "Caesar Salad", "price": 8.99, "description": "Fresh salad"}
            ],
            "metadata": {"scraping_source": "test", "scraping_method": "manual"}
        }

        self.db.store_restaurant_menu(menu_data)

        # Test search
        results = self.db.search_menu_items("pizza")
        assert len(results) >= 1
        assert "pizza" in results[0]["name"].lower()


class TestUtils:
    """Test utility functions"""

    def test_validate_url(self):
        """Test URL validation"""
        assert validate_url("https://www.example.com") == True
        assert validate_url("http://example.com") == True
        assert validate_url("not-a-url") == False
        assert validate_url("") == False

    def test_clean_price(self):
        """Test price cleaning"""
        assert clean_price("$12.99") == 12.99
        assert clean_price("€15,50") == 15.50
        assert clean_price("Price: $8.75") == 8.75
        assert clean_price("Free") is None
        assert clean_price("") is None

    def test_categorize_menu_item(self):
        """Test menu item categorization"""
        assert categorize_menu_item("Margherita Pizza") == "pizza"
        assert categorize_menu_item("Caesar Salad") == "salad"
        assert categorize_menu_item("Chicken Wings") == "chicken"
        assert categorize_menu_item("Random Item") == "other"

    def test_extract_dietary_info(self):
        """Test dietary information extraction"""
        text = "This is a vegetarian and gluten-free dish"
        dietary = extract_dietary_info(text)

        assert "vegetarian" in dietary
        assert "gluten-free" in dietary
        assert len(dietary) == 2


@pytest.fixture
def mock_selenium_driver():
    """Mock Selenium WebDriver for testing"""
    driver = Mock()
    driver.get = Mock()
    driver.find_element = Mock()
    driver.find_elements = Mock(return_value=[])
    driver.quit = Mock()
    return driver


class TestMenuScraper:
    """Test menu scraping functionality"""

    @patch('menu_scraper.webdriver.Chrome')
    def test_scraper_initialization(self, mock_chrome):
        """Test scraper initialization"""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        with MenuScraper(headless=True) as scraper:
            assert scraper.driver is not None

    def test_safe_element_extraction(self):
        """Test safe element text extraction"""
        scraper = MenuScraper()

        # Test with valid element
        mock_element = Mock()
        mock_element.text = "Test Text"
        assert scraper.extract_text_safely(mock_element) == "Test Text"

        # Test with None element
        assert scraper.extract_text_safely(None) == ""

        # Test with element that raises exception
        mock_element.text = Mock(side_effect=Exception("Error"))
        assert scraper.extract_text_safely(mock_element) == ""


class TestIntegration:
    """Integration tests"""

    def test_full_workflow_mock(self):
        """Test full workflow with mocked components"""
        # This would test the complete flow from scraping to storage
        # In a real scenario, you would mock the web requests and selenium
        pass


# Performance tests
class TestPerformance:
    """Performance tests for the scraping module"""

    def test_database_performance(self):
        """Test database performance with multiple items"""
        db = MenuDatabase("sqlite:///:memory:")

        # Create a large menu
        menu_items = []
        for i in range(100):
            menu_items.append({
                "name": f"Item {i}",
                "price": 10.0 + i,
                "description": f"Description for item {i}"
            })

        menu_data = {
            "restaurant_name": "Large Restaurant",
            "menu_items": menu_items,
            "metadata": {"scraping_source": "test", "scraping_method": "performance_test"}
        }

        import time
        start_time = time.time()
        restaurant_id = db.store_restaurant_menu(menu_data)
        end_time = time.time()

        assert restaurant_id is not None
        assert (end_time - start_time) < 5.0  # Should complete in under 5 seconds


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
