# Complete Backend System for Agentic AI Food Ordering Project
# Team Member B - Backend Core and Memory System
# Requirements: fastapi uvicorn databases[asyncpg] sqlalchemy pydantic faiss-cpu chromadb sentence-transformers numpy pandas

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

# FastAPI and database imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, validator
from databases import Database
import sqlalchemy
from sqlalchemy import create_engine, MetaData

# Vector embedding and similarity search
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost/agentic_food_ordering"
SYNC_DATABASE_URL = "postgresql://postgres:password@localhost/agentic_food_ordering"

database = Database(DATABASE_URL)
metadata = MetaData()
engine = create_engine(SYNC_DATABASE_URL)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Database Tables Definition
users = sqlalchemy.Table(
    "users", metadata,
    sqlalchemy.Column("user_id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("name", sqlalchemy.String(100), nullable=False),
    sqlalchemy.Column("email", sqlalchemy.String(100), unique=True, nullable=False),
    sqlalchemy.Column("phone", sqlalchemy.String(20), nullable=True),
    sqlalchemy.Column("preferences", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("dietary_restrictions", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("address", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
    sqlalchemy.Column("is_active", sqlalchemy.Boolean, default=True),
)

restaurants = sqlalchemy.Table(
    "restaurants", metadata,
    sqlalchemy.Column("restaurant_id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("name", sqlalchemy.String(200), nullable=False),
    sqlalchemy.Column("cuisine_type", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("location", sqlalchemy.String(500), nullable=True),
    sqlalchemy.Column("rating", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("contact_info", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("operating_hours", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("is_active", sqlalchemy.Boolean, default=True),
)

menu_items = sqlalchemy.Table(
    "menu_items", metadata,
    sqlalchemy.Column("item_id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("restaurant_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("restaurants.restaurant_id")),
    sqlalchemy.Column("name", sqlalchemy.String(200), nullable=False),
    sqlalchemy.Column("description", sqlalchemy.Text),
    sqlalchemy.Column("price", sqlalchemy.Numeric(10, 2), nullable=False),
    sqlalchemy.Column("category", sqlalchemy.String(100)),
    sqlalchemy.Column("ingredients", sqlalchemy.JSON),
    sqlalchemy.Column("nutritional_info", sqlalchemy.JSON),
    sqlalchemy.Column("allergens", sqlalchemy.JSON),
    sqlalchemy.Column("image_url", sqlalchemy.String(500)),
    sqlalchemy.Column("availability", sqlalchemy.Boolean, default=True),
    sqlalchemy.Column("preparation_time", sqlalchemy.Integer),  # in minutes
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

orders = sqlalchemy.Table(
    "orders", metadata,
    sqlalchemy.Column("order_id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.user_id"), nullable=False),
    sqlalchemy.Column("restaurant_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("restaurants.restaurant_id")),
    sqlalchemy.Column("order_items", sqlalchemy.JSON, nullable=False),  # List of {item_id, quantity, customizations}
    sqlalchemy.Column("total_amount", sqlalchemy.Numeric(10, 2), nullable=False),
    sqlalchemy.Column("order_status", sqlalchemy.String(50), default="pending"),
    sqlalchemy.Column("delivery_address", sqlalchemy.Text),
    sqlalchemy.Column("special_instructions", sqlalchemy.Text),
    sqlalchemy.Column("estimated_delivery_time", sqlalchemy.DateTime),
    sqlalchemy.Column("actual_delivery_time", sqlalchemy.DateTime),
    sqlalchemy.Column("payment_status", sqlalchemy.String(50), default="pending"),
    sqlalchemy.Column("payment_method", sqlalchemy.String(50)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

user_interactions = sqlalchemy.Table(
    "user_interactions", metadata,
    sqlalchemy.Column("interaction_id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.user_id")),
    sqlalchemy.Column("session_id", sqlalchemy.String(100)),
    sqlalchemy.Column("interaction_type", sqlalchemy.String(50)),  # search, view, order, feedback
    sqlalchemy.Column("item_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("menu_items.item_id")),
    sqlalchemy.Column("query_text", sqlalchemy.Text),
    sqlalchemy.Column("response_data", sqlalchemy.JSON),
    sqlalchemy.Column("user_rating", sqlalchemy.Integer),  # 1-5 scale
    sqlalchemy.Column("interaction_metadata", sqlalchemy.JSON),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

feedback = sqlalchemy.Table(
    "feedback", metadata,
    sqlalchemy.Column("feedback_id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.user_id")),
    sqlalchemy.Column("order_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("orders.order_id")),
    sqlalchemy.Column("item_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("menu_items.item_id")),
    sqlalchemy.Column("rating", sqlalchemy.Integer),  # 1-5 scale
    sqlalchemy.Column("comments", sqlalchemy.Text),
    sqlalchemy.Column("feedback_type", sqlalchemy.String(50)),  # food_quality, delivery, service
    sqlalchemy.Column("sentiment_score", sqlalchemy.Float),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

user_preferences = sqlalchemy.Table(
    "user_preferences", metadata,
    sqlalchemy.Column("preference_id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.user_id")),
    sqlalchemy.Column("preference_type", sqlalchemy.String(50)),  # cuisine, price_range, dietary
    sqlalchemy.Column("preference_value", sqlalchemy.String(200)),
    sqlalchemy.Column("weight", sqlalchemy.Float, default=1.0),  # Importance weight
    sqlalchemy.Column("learned_from_behavior", sqlalchemy.Boolean, default=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

# Create all tables
metadata.create_all(engine)

# Pydantic Models
class UserCreate(BaseModel):
    name: str = Field(..., max_length=100)
    email: EmailStr
    phone: Optional[str] = None
    preferences: Optional[Dict] = None
    dietary_restrictions: Optional[List[str]] = None
    address: Optional[str] = None

class UserResponse(BaseModel):
    user_id: int
    name: str
    email: str
    phone: Optional[str]
    preferences: Optional[Dict]
    dietary_restrictions: Optional[List[str]]
    address: Optional[str]
    created_at: datetime
    is_active: bool

class RestaurantCreate(BaseModel):
    name: str
    cuisine_type: Optional[str] = None
    location: Optional[str] = None
    rating: Optional[float] = None
    contact_info: Optional[Dict] = None
    operating_hours: Optional[Dict] = None

class RestaurantResponse(BaseModel):
    restaurant_id: int
    name: str
    cuisine_type: Optional[str]
    location: Optional[str]
    rating: Optional[float]
    contact_info: Optional[Dict]
    operating_hours: Optional[Dict]
    created_at: datetime
    is_active: bool

class MenuItemCreate(BaseModel):
    restaurant_id: int
    name: str
    description: Optional[str] = None
    price: float
    category: Optional[str] = None
    ingredients: Optional[List[str]] = None
    nutritional_info: Optional[Dict] = None
    allergens: Optional[List[str]] = None
    image_url: Optional[str] = None
    preparation_time: Optional[int] = None

class MenuItemResponse(BaseModel):
    item_id: int
    restaurant_id: int
    name: str
    description: Optional[str]
    price: float
    category: Optional[str]
    ingredients: Optional[List[str]]
    nutritional_info: Optional[Dict]
    allergens: Optional[List[str]]
    image_url: Optional[str]
    availability: bool
    preparation_time: Optional[int]
    created_at: datetime

class OrderCreate(BaseModel):
    user_id: int
    restaurant_id: int
    order_items: List[Dict]  # [{"item_id": 1, "quantity": 2, "customizations": {}}]
    delivery_address: Optional[str] = None
    special_instructions: Optional[str] = None
    payment_method: Optional[str] = None

class OrderResponse(BaseModel):
    order_id: int
    user_id: int
    restaurant_id: int
    order_items: List[Dict]
    total_amount: float
    order_status: str
    delivery_address: Optional[str]
    special_instructions: Optional[str]
    estimated_delivery_time: Optional[datetime]
    payment_status: str
    payment_method: Optional[str]
    created_at: datetime

class FeedbackCreate(BaseModel):
    user_id: int
    order_id: Optional[int] = None
    item_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None
    feedback_type: str = "general"

class InteractionLog(BaseModel):
    user_id: Optional[int] = None
    session_id: str
    interaction_type: str
    item_id: Optional[int] = None
    query_text: Optional[str] = None
    response_data: Optional[Dict] = None
    user_rating: Optional[int] = None
    interaction_metadata: Optional[Dict] = None

# Vector Memory System Class
class VectorMemorySystem:
    def __init__(self):
        self.faiss_index = None
        self.chroma_collection = None
        self.embeddings_cache = {}
        self.initialize_vector_stores()
    
    def initialize_vector_stores(self):
        """Initialize FAISS and ChromaDB collections"""
        try:
            # Initialize ChromaDB collection
            self.chroma_collection = chroma_client.get_or_create_collection(
                name="food_ordering_memory",
                metadata={"description": "User preferences and menu item embeddings"}
            )
            
            # Initialize FAISS index (will be built when data is available)
            self.faiss_index = None
            logger.info("Vector stores initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector stores: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        embedding = embedding_model.encode([text])[0]
        self.embeddings_cache[text] = embedding
        return embedding
    
    def add_menu_item_embedding(self, item_id: int, name: str, description: str, category: str):
        """Add menu item to vector store"""
        try:
            # Create combined text for embedding
            combined_text = f"{name} {description} {category}".strip()
            embedding = self.generate_embedding(combined_text)
            
            # Add to ChromaDB
            self.chroma_collection.add(
                embeddings=[embedding.tolist()],
                documents=[combined_text],
                metadatas=[{"item_id": item_id, "type": "menu_item"}],
                ids=[f"item_{item_id}"]
            )
            
            logger.info(f"Added menu item {item_id} to vector store")
        except Exception as e:
            logger.error(f"Error adding menu item to vector store: {e}")
    
    def add_user_preference_embedding(self, user_id: int, preferences: Dict):
        """Add user preferences to vector store"""
        try:
            # Create text from preferences
            pref_text = " ".join([f"{k}:{v}" for k, v in preferences.items() if v])
            embedding = self.generate_embedding(pref_text)
            
            # Add to ChromaDB
            self.chroma_collection.add(
                embeddings=[embedding.tolist()],
                documents=[pref_text],
                metadatas=[{"user_id": user_id, "type": "user_preference"}],
                ids=[f"user_{user_id}_pref"]
            )
            
            logger.info(f"Added user {user_id} preferences to vector store")
        except Exception as e:
            logger.error(f"Error adding user preferences to vector store: {e}")
    
    def find_similar_items(self, query: str, limit: int = 10) -> List[Dict]:
        """Find similar menu items based on query"""
        try:
            query_embedding = self.generate_embedding(query)
            
            # Query ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where={"type": "menu_item"}
            )
            
            similar_items = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similar_items.append({
                    "item_id": metadata['item_id'],
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "document": doc
                })
            
            return similar_items
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []

# Initialize vector memory system
vector_memory = VectorMemorySystem()

# FastAPI App
app = FastAPI(
    title="Agentic AI Food Ordering Backend",
    description="Complete backend system with vector memory and personalization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection events
@app.on_event("startup")
async def startup():
    await database.connect()
    logger.info("Database connected")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    logger.info("Database disconnected")

# Logging function
async def log_interaction(interaction: InteractionLog):
    """Log user interaction to database"""
    try:
        query = user_interactions.insert().values(
            user_id=interaction.user_id,
            session_id=interaction.session_id,
            interaction_type=interaction.interaction_type,
            item_id=interaction.item_id,
            query_text=interaction.query_text,
            response_data=interaction.response_data,
            user_rating=interaction.user_rating,
            interaction_metadata=interaction.interaction_metadata
        )
        await database.execute(query)
        logger.info(f"Logged interaction: {interaction.interaction_type}")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

# API Endpoints

# User Management
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, background_tasks: BackgroundTasks):
    """Create a new user"""
    try:
        query = users.insert().values(
            name=user.name,
            email=user.email,
            phone=user.phone,
            preferences=user.preferences,
            dietary_restrictions=user.dietary_restrictions,
            address=user.address
        )
        user_id = await database.execute(query)
        
        # Add user preferences to vector store
        if user.preferences:
            background_tasks.add_task(
                vector_memory.add_user_preference_embedding,
                user_id,
                user.preferences
            )
        
        # Log interaction
        background_tasks.add_task(
            log_interaction,
            InteractionLog(
                user_id=user_id,
                session_id=f"reg_{user_id}",
                interaction_type="user_registration",
                interaction_metadata={"email": user.email}
            )
        )
        
        return {**user.dict(), "user_id": user_id, "created_at": datetime.utcnow(), "is_active": True}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get user by ID"""
    query = users.select().where(users.c.user_id == user_id)
    user = await database.fetch_one(query)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/{user_id}/preferences")
async def update_user_preferences(user_id: int, preferences: Dict, background_tasks: BackgroundTasks):
    """Update user preferences"""
    query = users.update().where(users.c.user_id == user_id).values(
        preferences=preferences,
        updated_at=datetime.utcnow()
    )
    await database.execute(query)
    
    # Update vector store
    background_tasks.add_task(
        vector_memory.add_user_preference_embedding,
        user_id,
        preferences
    )
    
    return {"message": "Preferences updated successfully"}

# Restaurant Management
@app.post("/restaurants/", response_model=RestaurantResponse)
async def create_restaurant(restaurant: RestaurantCreate):
    """Create a new restaurant"""
    query = restaurants.insert().values(**restaurant.dict())
    restaurant_id = await database.execute(query)
    return {**restaurant.dict(), "restaurant_id": restaurant_id, "created_at": datetime.utcnow(), "is_active": True}

@app.get("/restaurants/", response_model=List[RestaurantResponse])
async def list_restaurants(cuisine_type: Optional[str] = None, location: Optional[str] = None):
    """List all restaurants with optional filters"""
    query = restaurants.select().where(restaurants.c.is_active == True)
    if cuisine_type:
        query = query.where(restaurants.c.cuisine_type.ilike(f"%{cuisine_type}%"))
    if location:
        query = query.where(restaurants.c.location.ilike(f"%{location}%"))
    
    return await database.fetch_all(query)

# Menu Item Management
@app.post("/menu_items/", response_model=MenuItemResponse)
async def create_menu_item(item: MenuItemCreate, background_tasks: BackgroundTasks):
    """Create a new menu item"""
    query = menu_items.insert().values(**item.dict())
    item_id = await database.execute(query)
    
    # Add to vector store
    background_tasks.add_task(
        vector_memory.add_menu_item_embedding,
        item_id,
        item.name,
        item.description or "",
        item.category or ""
    )
    
    return {**item.dict(), "item_id": item_id, "availability": True, "created_at": datetime.utcnow()}

@app.get("/menu_items/", response_model=List[MenuItemResponse])
async def list_menu_items(
    restaurant_id: Optional[int] = None,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    available_only: bool = True
):
    """List menu items with filters"""
    query = menu_items.select()
    
    if restaurant_id:
        query = query.where(menu_items.c.restaurant_id == restaurant_id)
    if category:
        query = query.where(menu_items.c.category.ilike(f"%{category}%"))
    if max_price:
        query = query.where(menu_items.c.price <= max_price)
    if available_only:
        query = query.where(menu_items.c.availability == True)
    
    return await database.fetch_all(query)

@app.get("/menu_items/search")
async def search_menu_items(
    query: str,
    user_id: Optional[int] = None,
    limit: int = 20,
    background_tasks: BackgroundTasks = None
):
    """Intelligent menu item search using vector similarity"""
    try:
        # Find similar items using vector search
        similar_items = vector_memory.find_similar_items(query, limit)
        
        if not similar_items:
            # Fallback to text search
            text_query = menu_items.select().where(
                sqlalchemy.or_(
                    menu_items.c.name.ilike(f"%{query}%"),
                    menu_items.c.description.ilike(f"%{query}%"),
                    menu_items.c.category.ilike(f"%{query}%")
                )
            ).limit(limit)
            results = await database.fetch_all(text_query)
        else:
            # Get full item details for similar items
            item_ids = [item['item_id'] for item in similar_items]
            detail_query = menu_items.select().where(menu_items.c.item_id.in_(item_ids))
            results = await database.fetch_all(detail_query)
            
            # Add similarity scores
            for result in results:
                for similar_item in similar_items:
                    if result['item_id'] == similar_item['item_id']:
                        result = dict(result)
                        result['similarity_score'] = similar_item['similarity_score']
        
        # Log search interaction
        if background_tasks and user_id:
            background_tasks.add_task(
                log_interaction,
                InteractionLog(
                    user_id=user_id,
                    session_id=f"search_{user_id}_{datetime.now().timestamp()}",
                    interaction_type="menu_search",
                    query_text=query,
                    response_data={"results_count": len(results)},
                    interaction_metadata={"search_method": "vector" if similar_items else "text"}
                )
            )
        
        return {"query": query, "results": results, "total_count": len(results)}
    
    except Exception as e:
        logger.error(f"Error in menu search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/menu_items/{item_id}", response_model=MenuItemResponse)
async def get_menu_item(item_id: int, user_id: Optional[int] = None, background_tasks: BackgroundTasks = None):
    """Get menu item by ID"""
    query = menu_items.select().where(menu_items.c.item_id == item_id)
    item = await database.fetch_one(query)
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    
    # Log view interaction
    if background_tasks and user_id:
        background_tasks.add_task(
            log_interaction,
            InteractionLog(
                user_id=user_id,
                session_id=f"view_{user_id}_{datetime.now().timestamp()}",
                interaction_type="item_view",
                item_id=item_id,
                interaction_metadata={"item_name": item['name']}
            )
        )
    
    return item

# Personalized Recommendations
@app.get("/users/{user_id}/recommendations")
async def get_personalized_recommendations(
    user_id: int,
    limit: int = 10,
    background_tasks: BackgroundTasks = None
):
    """Get personalized menu recommendations for user"""
    try:
        # Get user preferences
        user_query = users.select().where(users.c.user_id == user_id)
        user_data = await database.fetch_one(user_query)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        recommendations = []
        
        if user_data['preferences']:
            # Create query from preferences
            pref_text = " ".join([str(v) for v in user_data['preferences'].values() if v])
            similar_items = vector_memory.find_similar_items(pref_text, limit)
            
            if similar_items:
                item_ids = [item['item_id'] for item in similar_items]
                detail_query = menu_items.select().where(
                    sqlalchemy.and_(
                        menu_items.c.item_id.in_(item_ids),
                        menu_items.c.availability == True
                    )
                )
                recommendations = await database.fetch_all(detail_query)
                
                # Add recommendation scores
                for rec in recommendations:
                    for similar_item in similar_items:
                        if rec['item_id'] == similar_item['item_id']:
                            rec = dict(rec)
                            rec['recommendation_score'] = similar_item['similarity_score']
        
        # Fallback to popular items if no personalized recommendations
        if not recommendations:
            # Get most ordered items (simplified)
            popular_query = menu_items.select().where(
                menu_items.c.availability == True
            ).limit(limit)
            recommendations = await database.fetch_all(popular_query)
        
        # Log recommendation interaction
        if background_tasks:
            background_tasks.add_task(
                log_interaction,
                InteractionLog(
                    user_id=user_id,
                    session_id=f"rec_{user_id}_{datetime.now().timestamp()}",
                    interaction_type="recommendations_generated",
                    response_data={"recommendations_count": len(recommendations)},
                    interaction_metadata={"method": "personalized" if user_data['preferences'] else "popular"}
                )
            )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "method": "personalized" if user_data['preferences'] else "popular"
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

# Order Management
@app.post("/orders/", response_model=OrderResponse)
async def place_order(order: OrderCreate, background_tasks: BackgroundTasks):
    """Place a new order"""
    try:
        # Calculate total amount
        total_amount = 0
        for item in order.order_items:
            item_query = menu_items.select().where(menu_items.c.item_id == item['item_id'])
            menu_item = await database.fetch_one(item_query)
            if menu_item:
                total_amount += float(menu_item['price']) * item['quantity']
        
        # Create order
        query = orders.insert().values(
            user_id=order.user_id,
            restaurant_id=order.restaurant_id,
            order_items=order.order_items,
            total_amount=total_amount,
            delivery_address=order.delivery_address,
            special_instructions=order.special_instructions,
            payment_method=order.payment_method,
            estimated_delivery_time=datetime.utcnow() + timedelta(minutes=30)
        )
        order_id = await database.execute(query)
        
        # Log order interaction
        background_tasks.add_task(
            log_interaction,
            InteractionLog(
                user_id=order.user_id,
                session_id=f"order_{order.user_id}_{datetime.now().timestamp()}",
                interaction_type="order_placed",
                response_data={
                    "order_id": order_id,
                    "total_amount": float(total_amount),
                    "items_count": len(order.order_items)
                },
                interaction_metadata={"restaurant_id": order.restaurant_id}
            )
        )
        
        return {
            **order.dict(),
            "order_id": order_id,
            "total_amount": total_amount,
            "order_status": "pending",
            "payment_status": "pending",
            "estimated_delivery_time": datetime.utcnow() + timedelta(minutes=30),
            "created_at": datetime.utcnow()
        }
    
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail="Failed to place order")

@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: int):
    """Get order by ID"""
    query = orders.select().where(orders.c.order_id == order_id)
    order = await database.fetch_one(query)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@app.get("/users/{user_id}/orders", response_model=List[OrderResponse])
async def get_user_orders(user_id: int, limit: int = 50):
    """Get user's order history"""
    query = orders.select().where(orders.c.user_id == user_id).order_by(
        orders.c.created_at.desc()
    ).limit(limit)
    return await database.fetch_all(query)

@app.put("/orders/{order_id}/status")
async def update_order_status(order_id: int, status: str, background_tasks: BackgroundTasks):
    """Update order status"""
    valid_statuses = ["pending", "confirmed", "preparing", "ready", "delivered", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    update_data = {"order_status": status, "updated_at": datetime.utcnow()}
    if status == "delivered":
        update_data["actual_delivery_time"] = datetime.utcnow()
    
    query = orders.update().where(orders.c.order_id == order_id).values(**update_data)
    await database.execute(query)
    
    # Log status update
    order_query = orders.select().where(orders.c.order_id == order_id)
    order_data = await database.fetch_one(order_query)
    if order_data:
        background_tasks.add_task(
            log_interaction,
            InteractionLog(
                user_id=order_data['user_id'],
                session_id=f"status_{order_id}_{datetime.now().timestamp()}",
                interaction_type="order_status_updated",
                response_data={"order_id": order_id, "new_status": status},
                interaction_metadata={"previous_status": order_data['order_status']}
            )
        )
    
    return {"message": f"Order status updated to {status}"}

# Feedback System
@app.post("/feedback/")
async def submit_feedback(feedback_data: FeedbackCreate, background_tasks: BackgroundTasks):
    """Submit feedback"""
    try:
        # Calculate sentiment score (simplified)
        sentiment_score = 0.0
        if feedback_data.comments:
            # Simple sentiment calculation based on rating
            sentiment_score = (feedback_data.rating - 3) / 2  # Normalize to -1 to 1
        
        query = feedback.insert().values(
            user_id=feedback_data.user_id,
            order_id=feedback_data.order_id,
            item_id=feedback_data.item_id,
            rating=feedback_data.rating,
            comments=feedback_data.comments,
            feedback_type=feedback_data.feedback_type,
            sentiment_score=sentiment_score
        )
        feedback_id = await database.execute(query)
        
        # Log feedback interaction
        background_tasks.add_task(
            log_interaction,
            InteractionLog(
                user_id=feedback_data.user_id,
                session_id=f"feedback_{feedback_data.user_id}_{datetime.now().timestamp()}",
                interaction_type="feedback_submitted",
                user_rating=feedback_data.rating,
                response_data={"feedback_id": feedback_id, "sentiment_score": sentiment_score},
                interaction_metadata={"feedback_type": feedback_data.feedback_type}
            )
        )
        
        return {
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully",
            "sentiment_score": sentiment_score
        }
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

# Analytics and Insights
@app.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: int):
    """Get user behavior analytics"""
    try:
        # Get interaction summary
        interaction_query = """
        SELECT interaction_type, COUNT(*) as count
        FROM user_interactions 
        WHERE user_id = :user_id 
        GROUP BY interaction_type
        """
        interactions = await database.fetch_all(
            sqlalchemy.text(interaction_query), 
            {"user_id": user_id}
        )
        
        # Get order summary
        order_query = """
        SELECT COUNT(*) as total_orders, 
               AVG(total_amount) as avg_order_value,
               SUM(total_amount) as total_spent
        FROM orders 
        WHERE user_id = :user_id
        """
        order_stats = await database.fetch_one(
            sqlalchemy.text(order_query), 
            {"user_id": user_id}
        )
        
        # Get feedback summary
        feedback_query = """
        SELECT AVG(rating) as avg_rating, COUNT(*) as feedback_count
        FROM feedback 
        WHERE user_id = :user_id
        """
        feedback_stats = await database.fetch_one(
            sqlalchemy.text(feedback_query), 
            {"user_id": user_id}
        )
        
        return {
            "user_id": user_id,
            "interactions": dict(interactions) if interactions else {},
            "order_statistics": dict(order_stats) if order_stats else {},
            "feedback_statistics": dict(feedback_stats) if feedback_stats else {}
        }
    
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@app.get("/analytics/popular_items")
async def get_popular_items(limit: int = 10):
    """Get most popular menu items"""
    try:
        # This is a simplified version - in production, you'd want more complex analytics
        query = """
        SELECT mi.item_id, mi.name, mi.category, COUNT(ui.item_id) as interaction_count
        FROM menu_items mi
        LEFT JOIN user_interactions ui ON mi.item_id = ui.item_id
        WHERE ui.interaction_type IN ('item_view', 'order_placed')
        GROUP BY mi.item_id, mi.name, mi.category
        ORDER BY interaction_count DESC
        LIMIT :limit
        """
        popular_items = await database.fetch_all(
            sqlalchemy.text(query), 
            {"limit": limit}
        )
        
        return {"popular_items": [dict(item) for item in popular_items]}
    
    except Exception as e:
        logger.error(f"Error getting popular items: {e}")
        raise HTTPException(status_code=500, detail="Failed to get popular items")

# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        await database.fetch_one("SELECT 1")
        
        # Test vector store
        test_embedding = vector_memory.generate_embedding("test")
        
        return {
            "status": "healthy",
            "database": "connected",
            "vector_store": "operational",
            "embedding_model": "loaded",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

# Export data endpoints (for data analysis)
@app.get("/export/interactions")
async def export_interactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[int] = None
):
    """Export interaction data for analysis"""
    try:
        query = user_interactions.select()
        
        if start_date:
            query = query.where(user_interactions.c.created_at >= start_date)
        if end_date:
            query = query.where(user_interactions.c.created_at <= end_date)
        if user_id:
            query = query.where(user_interactions.c.user_id == user_id)
        
        interactions = await database.fetch_all(query)
        return {"interactions": [dict(interaction) for interaction in interactions]}
    
    except Exception as e:
        logger.error(f"Error exporting interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)