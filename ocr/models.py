"""Data models for menu scraping."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from enum import Enum

Base = declarative_base()

class ScrapingStatus(str, Enum):
    """Status of scraping jobs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MenuItemType(str, Enum):
    """Types of menu items."""
    APPETIZER = "appetizer"
    MAIN_COURSE = "main_course"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    SIDE_DISH = "side_dish"
    COMBO = "combo"
    OTHER = "other"

# SQLAlchemy Models
class Restaurant(Base):
    """Restaurant database model."""
    __tablename__ = "restaurants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    url = Column(Text, nullable=False)
    cuisine_type = Column(String(100))
    rating = Column(Float)
    address = Column(Text)
    phone = Column(String(20))
    delivery_time = Column(String(50))
    minimum_order = Column(Float)
    delivery_fee = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    menu_sections = relationship("MenuSection", back_populates="restaurant", cascade="all, delete-orphan")
    scraping_jobs = relationship("ScrapingJob", back_populates="restaurant")

class MenuSection(Base):
    """Menu section database model."""
    __tablename__ = "menu_sections"
    
    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    display_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    restaurant = relationship("Restaurant", back_populates="menu_sections")
    menu_items = relationship("MenuItem", back_populates="section", cascade="all, delete-orphan")

class MenuItem(Base):
    """Menu item database model."""
    __tablename__ = "menu_items"
    
    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(Integer, ForeignKey("menu_sections.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(Float)
    original_price = Column(Float)  # For discounted items
    currency = Column(String(5), default="USD")
    item_type = Column(String(50))
    is_vegetarian = Column(Boolean, default=False)
    is_vegan = Column(Boolean, default=False)
    is_spicy = Column(Boolean, default=False)
    allergens = Column(JSON)  # List of allergens
    nutritional_info = Column(JSON)
    image_url = Column(Text)
    availability = Column(Boolean, default=True)
    preparation_time = Column(String(20))
    calories = Column(Integer)
    display_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    section = relationship("MenuSection", back_populates="menu_items")

class ScrapingJob(Base):
    """Scraping job tracking model."""
    __tablename__ = "scraping_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"))
    url = Column(Text, nullable=False)
    platform = Column(String(50), nullable=False)
    status = Column(String(20), default=ScrapingStatus.PENDING)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    items_scraped = Column(Integer, default=0)
    sections_scraped = Column(Integer, default=0)
    metadata = Column(JSON)  # Additional job information
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    restaurant = relationship("Restaurant", back_populates="scraping_jobs")

# Pydantic Models for API
class MenuItemBase(BaseModel):
    """Base menu item schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    price: Optional[float] = Field(None, ge=0)
    original_price: Optional[float] = Field(None, ge=0)
    currency: str = Field(default="USD", max_length=5)
    item_type: Optional[MenuItemType] = MenuItemType.OTHER
    is_vegetarian: bool = False
    is_vegan: bool = False
    is_spicy: bool = False
    allergens: Optional[List[str]] = None
    nutritional_info: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    availability: bool = True
    preparation_time: Optional[str] = None
    calories: Optional[int] = Field(None, ge=0)
    display_order: int = 0

class MenuItemCreate(MenuItemBase):
    """Menu item creation schema."""
    pass

class MenuItemUpdate(BaseModel):
    """Menu item update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    price: Optional[float] = Field(None, ge=0)
    original_price: Optional[float] = Field(None, ge=0)
    currency: Optional[str] = Field(None, max_length=5)
    item_type: Optional[MenuItemType] = None
    is_vegetarian: Optional[bool] = None
    is_vegan: Optional[bool] = None
    is_spicy: Optional[bool] = None
    allergens: Optional[List[str]] = None
    nutritional_info: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    availability: Optional[bool] = None
    preparation_time: Optional[str] = None
    calories: Optional[int] = Field(None, ge=0)
    display_order: Optional[int] = None

class MenuItemResponse(MenuItemBase):
    """Menu item response schema."""
    id: int
    section_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class MenuSectionBase(BaseModel):
    """Base menu section schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    display_order: int = 0

class MenuSectionCreate(MenuSectionBase):
    """Menu section creation schema."""
    pass

class MenuSectionResponse(MenuSectionBase):
    """Menu section response schema."""
    id: int
    restaurant_id: int
    created_at: datetime
    menu_items: List[MenuItemResponse] = []
    
    class Config:
        from_attributes = True

class RestaurantBase(BaseModel):
    """Base restaurant schema."""
    name: str = Field(..., min_length=1, max_length=255)
    platform: str = Field(..., max_length=50)
    url: str
    cuisine_type: Optional[str] = Field(None, max_length=100)
    rating: Optional[float] = Field(None, ge=0, le=5)
    address: Optional[str] = None
    phone: Optional[str] = Field(None, max_length=20)
    delivery_time: Optional[str] = Field(None, max_length=50)
    minimum_order: Optional[float] = Field(None, ge=0)
    delivery_fee: Optional[float] = Field(None, ge=0)

class RestaurantCreate(RestaurantBase):
    """Restaurant creation schema."""
    pass

class RestaurantResponse(RestaurantBase):
    """Restaurant response schema."""
    id: int
    created_at: datetime
    updated_at: datetime
    menu_sections: List[MenuSectionResponse] = []
    
    class Config:
        from_attributes = True

class ScrapingJobCreate(BaseModel):
    """Scraping job creation schema."""
    url: str
    platform: str = Field(..., max_length=50)
    metadata: Optional[Dict[str, Any]] = None

class ScrapingJobResponse(BaseModel):
    """Scraping job response schema."""
    id: int
    restaurant_id: Optional[int]
    url: str
    platform: str
    status: ScrapingStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    items_scraped: int
    sections_scraped: int
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True

class ScrapingResult(BaseModel):
    """Complete scraping result schema."""
    restaurant: RestaurantResponse
    total_items: int
    total_sections: int
    processing_time: float
    warnings: List[str] = []

class OCRRequest(BaseModel):
    """OCR processing request schema."""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    language: str = Field(default="eng", max_length=10)
    preprocessing: bool = True

class OCRResponse(BaseModel):
    """OCR processing response schema."""
    extracted_text: str
    confidence: Optional[float] = None
    processing_time: float
    warnings: List[str] = []
