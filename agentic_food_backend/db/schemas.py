# Complete Pydantic schemas for the agentic food backend
# agentic_food_backend/db/schemas.py

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Enums for validation
class OrderStatus(str, Enum):
    pending = "pending"
    confirmed = "confirmed"
    preparing = "preparing"
    ready = "ready"
    delivered = "delivered"
    cancelled = "cancelled"

class PaymentStatus(str, Enum):
    pending = "pending"
    paid = "paid"
    failed = "failed"
    refunded = "refunded"

class FeedbackType(str, Enum):
    general = "general"
    food_quality = "food_quality"
    delivery = "delivery"
    service = "service"

class InteractionType(str, Enum):
    menu_search = "menu_search"
    item_view = "item_view"
    order_placed = "order_placed"
    feedback_submitted = "feedback_submitted"
    user_registration = "user_registration"
    recommendations_generated = "recommendations_generated"
    order_status_updated = "order_status_updated"

# Base schemas
class UserBase(BaseModel):
    name: str = Field(..., max_length=100, min_length=1)
    email: EmailStr
    phone: Optional[str] = Field(None, max_length=20)
    preferences: Optional[Dict[str, Any]] = None
    dietary_restrictions: Optional[List[str]] = None
    address: Optional[str] = Field(None, max_length=500)

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

class UserResponse(UserBase):
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool = True

    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100, min_length=1)
    phone: Optional[str] = Field(None, max_length=20)
    preferences: Optional[Dict[str, Any]] = None
    dietary_restrictions: Optional[List[str]] = None
    address: Optional[str] = Field(None, max_length=500)

# Restaurant schemas
class RestaurantBase(BaseModel):
    name: str = Field(..., max_length=200, min_length=1)
    cuisine_type: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=500)
    rating: Optional[float] = Field(None, ge=0, le=5)
    contact_info: Optional[Dict[str, Any]] = None
    operating_hours: Optional[Dict[str, Any]] = None

class RestaurantCreate(RestaurantBase):
    pass

class RestaurantResponse(RestaurantBase):
    restaurant_id: int
    created_at: datetime
    is_active: bool = True

    class Config:
        orm_mode = True

class RestaurantUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200, min_length=1)
    cuisine_type: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=500)
    rating: Optional[float] = Field(None, ge=0, le=5)
    contact_info: Optional[Dict[str, Any]] = None
    operating_hours: Optional[Dict[str, Any]] = None

# Menu Item schemas
class MenuItemBase(BaseModel):
    restaurant_id: int = Field(..., gt=0)
    name: str = Field(..., max_length=200, min_length=1)
    description: Optional[str] = None
    price: float = Field(..., gt=0, le=99999.99)
    category: Optional[str] = Field(None, max_length=100)
    ingredients: Optional[List[str]] = None
    nutritional_info: Optional[Dict[str, Any]] = None
    allergens: Optional[List[str]] = None
    image_url: Optional[str] = Field(None, max_length=500)
    preparation_time: Optional[int] = Field(None, ge=0, le=1440)  # Max 24 hours

class MenuItemCreate(MenuItemBase):
    pass

class MenuItemResponse(MenuItemBase):
    item_id: int
    availability: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class MenuItemUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200, min_length=1)
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0, le=99999.99)
    category: Optional[str] = Field(None, max_length=100)
    ingredients: Optional[List[str]] = None
    nutritional_info: Optional[Dict[str, Any]] = None
    allergens: Optional[List[str]] = None
    image_url: Optional[str] = Field(None, max_length=500)
    preparation_time: Optional[int] = Field(None, ge=0, le=1440)
    availability: Optional[bool] = None

# Order schemas
class OrderItemSchema(BaseModel):
    item_id: int = Field(..., gt=0)
    quantity: int = Field(..., gt=0, le=100)
    customizations: Optional[Dict[str, Any]] = None

class OrderBase(BaseModel):
    user_id: int = Field(..., gt=0)
    restaurant_id: int = Field(..., gt=0)
    order_items: List[OrderItemSchema] = Field(..., min_items=1)
    delivery_address: Optional[str] = Field(None, max_length=500)
    special_instructions: Optional[str] = Field(None, max_length=1000)
    payment_method: Optional[str] = Field(None, max_length=50)

class OrderCreate(OrderBase):
    @validator('order_items')
    def validate_order_items(cls, v):
        if not v:
            raise ValueError('Order must contain at least one item')
        return v

class OrderResponse(OrderBase):
    order_id: int
    total_amount: float
    order_status: OrderStatus = OrderStatus.pending
    payment_status: PaymentStatus = PaymentStatus.pending
    estimated_delivery_time: Optional[datetime] = None
    actual_delivery_time: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class OrderUpdate(BaseModel):
    order_status: Optional[OrderStatus] = None
    payment_status: Optional[PaymentStatus] = None
    estimated_delivery_time: Optional[datetime] = None
    actual_delivery_time: Optional[datetime] = None

# Feedback schemas
class FeedbackBase(BaseModel):
    user_id: int = Field(..., gt=0)
    order_id: Optional[int] = Field(None, gt=0)
    item_id: Optional[int] = Field(None, gt=0)
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=2000)
    feedback_type: FeedbackType = FeedbackType.general

class FeedbackCreate(FeedbackBase):
    @validator('order_id', 'item_id')
    def validate_feedback_target(cls, v, values):
        order_id = values.get('order_id')
        item_id = values.get('item_id')
        if not order_id and not item_id:
            raise ValueError('Feedback must be for either an order or menu item')
        return v

class FeedbackResponse(FeedbackBase):
    feedback_id: int
    sentiment_score: Optional[float] = None
    created_at: datetime

    class Config:
        orm_mode = True

# Interaction schemas
class InteractionBase(BaseModel):
    user_id: Optional[int] = Field(None, gt=0)
    session_id: str = Field(..., max_length=100)
    interaction_type: InteractionType
    item_id: Optional[int] = Field(None, gt=0)
    query_text: Optional[str] = Field(None, max_length=1000)
    response_data: Optional[Dict[str, Any]] = None
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    interaction_metadata: Optional[Dict[str, Any]] = None

class InteractionLog(InteractionBase):
    pass

class InteractionResponse(InteractionBase):
    interaction_id: int
    created_at: datetime

    class Config:
        orm_mode = True

# Authentication schemas
class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None

class TokenRefresh(BaseModel):
    refresh_token: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Search and recommendation schemas
class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    user_id: Optional[int] = Field(None, gt=0)
    limit: int = Field(20, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_count: int
    method: str = "text"  # or "vector"
    filters_applied: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    total_count: int
    method: str = "popular"  # or "personalized"
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# Analytics schemas
class UserAnalytics(BaseModel):
    user_id: int
    interaction_summary: Dict[str, int]
    order_statistics: Dict[str, Any]
    feedback_statistics: Dict[str, Any]
    recent_activity: List[Dict[str, Any]]

class PopularItems(BaseModel):
    popular_items: List[Dict[str, Any]]
    period: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class FeedbackAnalytics(BaseModel):
    overall_stats: Dict[str, Any]
    feedback_by_type: List[Dict[str, Any]]
    recent_trends: List[Dict[str, Any]]

# Error schemas
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ValidationErrorResponse(BaseModel):
    detail: str
    errors: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Health check schema
class HealthCheck(BaseModel):
    status: str
    database: str
    vector_store: str
    embedding_model: str
    timestamp: datetime