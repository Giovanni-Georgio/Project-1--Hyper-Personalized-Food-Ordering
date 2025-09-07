from sqlalchemy import (
    Table, Column, Integer, String, Boolean, DateTime, JSON, Numeric,
    ForeignKey, Float, Text
)
from agentic_food_backend.db.database import metadata
from datetime import datetime

users = Table(
    "users", metadata,
    Column("user_id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(100), nullable=False),
    Column("email", String(100), unique=True, nullable=False),
    Column("phone", String(20), nullable=True),
    Column("preferences", JSON, nullable=True),
    Column("dietary_restrictions", JSON, nullable=True),
    Column("address", Text, nullable=True),
    Column("password_hash", String(256), nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
    Column("is_active", Boolean, default=True)
)

restaurants = Table(
    "restaurants", metadata,
    Column("restaurant_id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(200), nullable=False),
    Column("cuisine_type", String(100), nullable=True),
    Column("location", String(500), nullable=True),
    Column("rating", Float, nullable=True),
    Column("contact_info", JSON, nullable=True),
    Column("operating_hours", JSON, nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("is_active", Boolean, default=True)
)

menu_items = Table(
    "menu_items", metadata,
    Column("item_id", Integer, primary_key=True, autoincrement=True),
    Column("restaurant_id", Integer, ForeignKey("restaurants.restaurant_id")),
    Column("name", String(200), nullable=False),
    Column("description", Text),
    Column("price", Numeric(10, 2), nullable=False),
    Column("category", String(100)),
    Column("ingredients", JSON),
    Column("nutritional_info", JSON),
    Column("allergens", JSON),
    Column("image_url", String(500)),
    Column("availability", Boolean, default=True),
    Column("preparation_time", Integer),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

orders = Table(
    "orders", metadata,
    Column("order_id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.user_id"), nullable=False),
    Column("restaurant_id", Integer, ForeignKey("restaurants.restaurant_id")),
    Column("order_items", JSON, nullable=False),
    Column("total_amount", Numeric(10, 2), nullable=False),
    Column("order_status", String(50), default="pending"),
    Column("delivery_address", Text),
    Column("special_instructions", Text),
    Column("estimated_delivery_time", DateTime),
    Column("actual_delivery_time", DateTime),
    Column("payment_status", String(50), default="pending"),
    Column("payment_method", String(50)),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

user_interactions = Table(
    "user_interactions", metadata,
    Column("interaction_id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.user_id")),
    Column("session_id", String(100)),
    Column("interaction_type", String(50)),
    Column("item_id", Integer, ForeignKey("menu_items.item_id")),
    Column("query_text", Text),
    Column("response_data", JSON),
    Column("user_rating", Integer),
    Column("interaction_metadata", JSON),
    Column("created_at", DateTime, default=datetime.utcnow),
)

feedback = Table(
    "feedback", metadata,
    Column("feedback_id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.user_id")),
    Column("order_id", Integer, ForeignKey("orders.order_id")),
    Column("item_id", Integer, ForeignKey("menu_items.item_id")),
    Column("rating", Integer),
    Column("comments", Text),
    Column("feedback_type", String(50)),
    Column("sentiment_score", Float),
    Column("created_at", DateTime, default=datetime.utcnow),
)

user_preferences = Table(
    "user_preferences", metadata,
    Column("preference_id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.user_id")),
    Column("preference_type", String(50)),
    Column("preference_value", String(200)),
    Column("weight", Float, default=1.0),
    Column("learned_from_behavior", Boolean, default=False),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)
