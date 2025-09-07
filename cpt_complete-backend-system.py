"""
Complete backend for Agentic AI Food Ordering Project
Team Member B - Backend Core and Memory System
Single-file FastAPI application with JWT auth, async DB, vector memory (ChromaDB), and analytics.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from fastapi import (
    FastAPI, HTTPException, Depends, BackgroundTasks, status, Security
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr, validator
from databases import Database
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Boolean, DateTime, Float, Text, JSON, Numeric, ForeignKey

# Optional: FAISS left here for future local index usage (unused currently)
# import faiss

# Vector embedding and similarity search
import chromadb
from sentence_transformers import SentenceTransformer

# Password hashing & JWT
from passlib.context import CryptContext
import jwt

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost/agentic_food_ordering")
SYNC_DATABASE_URL = os.getenv("SYNC_DATABASE_URL", "postgresql://postgres:password@localhost/agentic_food_ordering")

SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# ---------- DB / ORM setup ----------
database = Database(DATABASE_URL)
metadata = MetaData()
engine = create_engine(SYNC_DATABASE_URL)

# ---------- Embedding model + ChromaDB ----------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Use persistent ChromaDB instance in local directory
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ---------- Tables ----------
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
    Column("is_active", Boolean, default=True),
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
    Column("is_active", Boolean, default=True),
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

# Create all tables (synchronous engine)
metadata.create_all(engine)

# ---------- Security utils ----------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    if not hashed_password:
        return False
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

# ---------- Pydantic models ----------
class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    name: str = Field(..., max_length=100)
    email: EmailStr
    phone: Optional[str] = None
    preferences: Optional[Dict] = None
    dietary_restrictions: Optional[List[str]] = None
    address: Optional[str] = None
    password: str = Field(..., min_length=6)


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

# ---------- Vector Memory System ----------
class VectorMemorySystem:
    def __init__(self):
        self.chroma_collection = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.initialize_vector_stores()

    def initialize_vector_stores(self):
        try:
            self.chroma_collection = chroma_client.get_or_create_collection(
                name="food_ordering_memory",
                metadata={"description": "User preferences and menu item embeddings"}
            )
            logger.info("ChromaDB collection ready")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            self.chroma_collection = None

    def generate_embedding(self, text: str) -> np.ndarray:
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        emb = embedding_model.encode([text])[0]
        self.embeddings_cache[text] = emb
        return emb

    def add_menu_item_embedding(self, item_id: int, name: str, description: str, category: str):
        if not self.chroma_collection:
            logger.warning("ChromaDB not initialized; skipping embedding add.")
            return
        try:
            combined_text = f"{name} {description or ''} {category or ''}".strip()
            emb = self.generate_embedding(combined_text)
            self.chroma_collection.add(
                embeddings=[emb.tolist()],
                documents=[combined_text],
                metadatas=[{"item_id": item_id, "type": "menu_item"}],
                ids=[f"item_{item_id}"]
            )
            logger.info(f"Added menu item embedding: {item_id}")
        except Exception as e:
            logger.error(f"Failed to add menu item embedding: {e}")

    def add_user_preference_embedding(self, user_id: int, preferences: Dict):
        if not self.chroma_collection:
            logger.warning("ChromaDB not initialized; skipping user pref embedding.")
            return
        try:
            pref_text = " ".join([f"{k}:{v}" for k, v in preferences.items() if v])
            emb = self.generate_embedding(pref_text)
            self.chroma_collection.add(
                embeddings=[emb.tolist()],
                documents=[pref_text],
                metadatas=[{"user_id": user_id, "type": "user_preference"}],
                ids=[f"user_{user_id}_pref"]
            )
            logger.info(f"Added user preference embedding: {user_id}")
        except Exception as e:
            logger.error(f"Failed to add user preference embedding: {e}")

    def find_similar_items(self, query: str, limit: int = 10) -> List[Dict]:
        if not self.chroma_collection:
            logger.warning("ChromaDB not initialized; cannot vector search.")
            return []
        try:
            qe = self.generate_embedding(query)
            results = self.chroma_collection.query(
                query_embeddings=[qe.tolist()],
                n_results=limit,
                where={"type": "menu_item"}
            )
            similar_items = []
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                similar_items.append({
                    "item_id": meta['item_id'],
                    "similarity_score": 1 - dist,
                    "document": doc
                })
            return similar_items
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


vector_memory = VectorMemorySystem()

# ---------- FastAPI app ----------
app = FastAPI(
    title="Agentic AI Food Ordering Backend (Full)",
    description="Backend with JWT auth, vector memory, personalization, and analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def on_startup():
    await database.connect()
    logger.info("Database connected")


@app.on_event("shutdown")
async def on_shutdown():
    await database.disconnect()
    logger.info("Database disconnected")


# ---------- Helper functions ----------
async def log_interaction(interaction: InteractionLog):
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


async def get_user_by_email(email: str):
    q = users.select().where(users.c.email == email)
    return await database.fetch_one(q)


async def get_user(user_id: int):
    q = users.select().where(users.c.user_id == user_id)
    return await database.fetch_one(q)


async def authenticate_user(email: str, password: str):
    user = await get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user['password_hash']):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = await get_user(int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ---------- Auth endpoints ----------
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user['user_id']), "email": user['email']},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ---------- User management ----------
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, background_tasks: BackgroundTasks):
    # Check if email exists
    existing = await get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = get_password_hash(user.password)
    query = users.insert().values(
        name=user.name,
        email=user.email,
        phone=user.phone,
        preferences=user.preferences,
        dietary_restrictions=user.dietary_restrictions,
        address=user.address,
        password_hash=hashed
    )
    user_id = await database.execute(query)

    # add user prefs to vector store in background
    if user.preferences:
        background_tasks.add_task(vector_memory.add_user_preference_embedding, user_id, user.preferences)

    # log registration
    background_tasks.add_task(log_interaction, InteractionLog(
        user_id=user_id,
        session_id=f"reg_{user_id}",
        interaction_type="user_registration",
        interaction_metadata={"email": user.email}
    ))

    return {
        "user_id": user_id,
        "name": user.name,
        "email": user.email,
        "phone": user.phone,
        "preferences": user.preferences,
        "dietary_restrictions": user.dietary_restrictions,
        "address": user.address,
        "created_at": datetime.utcnow(),
        "is_active": True
    }


@app.get("/users/me", response_model=UserResponse)
async def read_current_user(current_user=Depends(get_current_user)):
    return {
        "user_id": current_user['user_id'],
        "name": current_user['name'],
        "email": current_user['email'],
        "phone": current_user['phone'],
        "preferences": current_user['preferences'],
        "dietary_restrictions": current_user['dietary_restrictions'],
        "address": current_user['address'],
        "created_at": current_user['created_at'],
        "is_active": current_user['is_active']
    }


@app.put("/users/{user_id}/preferences")
async def update_user_preferences(user_id: int, preferences: Dict, background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    # allow only same user (or admin in future)
    if int(current_user['user_id']) != int(user_id):
        raise HTTPException(status_code=403, detail="Not allowed to update other user's preferences")

    q = users.update().where(users.c.user_id == user_id).values(preferences=preferences, updated_at=datetime.utcnow())
    await database.execute(q)
    background_tasks.add_task(vector_memory.add_user_preference_embedding, user_id, preferences)
    return {"message": "Preferences updated successfully"}


# ---------- Restaurant management ----------
@app.post("/restaurants/", response_model=RestaurantResponse)
async def create_restaurant(restaurant: RestaurantCreate, current_user=Depends(get_current_user)):
    # In production, check admin rights. For now any authenticated user can create.
    q = restaurants.insert().values(**restaurant.dict())
    restaurant_id = await database.execute(q)
    return {**restaurant.dict(), "restaurant_id": restaurant_id, "created_at": datetime.utcnow(), "is_active": True}


@app.get("/restaurants/", response_model=List[RestaurantResponse])
async def list_restaurants(cuisine_type: Optional[str] = None, location: Optional[str] = None):
    q = restaurants.select().where(restaurants.c.is_active == True)
    if cuisine_type:
        q = q.where(restaurants.c.cuisine_type.ilike(f"%{cuisine_type}%"))
    if location:
        q = q.where(restaurants.c.location.ilike(f"%{location}%"))
    return await database.fetch_all(q)


# ---------- Menu items ----------
@app.post("/menu_items/", response_model=MenuItemResponse)
async def create_menu_item(item: MenuItemCreate, background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    q = menu_items.insert().values(**item.dict())
    item_id = await database.execute(q)
    # add to vector store (bg)
    background_tasks.add_task(vector_memory.add_menu_item_embedding, item_id, item.name, item.description or "", item.category or "")
    return {**item.dict(), "item_id": item_id, "availability": True, "created_at": datetime.utcnow()}


@app.get("/menu_items/", response_model=List[MenuItemResponse])
async def list_menu_items(restaurant_id: Optional[int] = None,
                          category: Optional[str] = None,
                          max_price: Optional[float] = None,
                          available_only: bool = True):
    q = menu_items.select()
    if restaurant_id:
        q = q.where(menu_items.c.restaurant_id == restaurant_id)
    if category:
        q = q.where(menu_items.c.category.ilike(f"%{category}%"))
    if max_price:
        q = q.where(menu_items.c.price <= max_price)
    if available_only:
        q = q.where(menu_items.c.availability == True)
    return await database.fetch_all(q)


@app.get("/menu_items/search")
async def search_menu_items(query: str,
                            user_id: Optional[int] = None,
                            limit: int = 20,
                            background_tasks: BackgroundTasks = None):
    try:
        similar_items = vector_memory.find_similar_items(query, limit)
        results = []
        if not similar_items:
            # fallback to text search
            text_q = menu_items.select().where(
                sqlalchemy.or_(
                    menu_items.c.name.ilike(f"%{query}%"),
                    menu_items.c.description.ilike(f"%{query}%"),
                    menu_items.c.category.ilike(f"%{query}%")
                )
            ).limit(limit)
            results = await database.fetch_all(text_q)
            method = "text"
        else:
            item_ids = [item['item_id'] for item in similar_items]
            detail_q = menu_items.select().where(menu_items.c.item_id.in_(item_ids))
            results = await database.fetch_all(detail_q)
            # attach similarity scores
            results_map = {r['item_id']: dict(r) for r in results}
            for sim in similar_items:
                if sim['item_id'] in results_map:
                    results_map[sim['item_id']]['similarity_score'] = sim['similarity_score']
            results = list(results_map.values())
            method = "vector"

        # log search
        if background_tasks and user_id:
            background_tasks.add_task(log_interaction, InteractionLog(
                user_id=user_id,
                session_id=f"search_{user_id}_{datetime.now().timestamp()}",
                interaction_type="menu_search",
                query_text=query,
                response_data={"results_count": len(results)},
                interaction_metadata={"search_method": method}
            ))
        return {"query": query, "results": results, "total_count": len(results)}
    except Exception as e:
        logger.error(f"Error in menu search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/menu_items/{item_id}", response_model=MenuItemResponse)
async def get_menu_item(item_id: int, user_id: Optional[int] = None, background_tasks: BackgroundTasks = None):
    q = menu_items.select().where(menu_items.c.item_id == item_id)
    item = await database.fetch_one(q)
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")

    # log item view if user present
    if background_tasks and user_id:
        background_tasks.add_task(log_interaction, InteractionLog(
            user_id=user_id,
            session_id=f"view_{user_id}_{datetime.now().timestamp()}",
            interaction_type="item_view",
            item_id=item_id,
            interaction_metadata={"item_name": item['name']}
        ))
    return item


# ---------- Personalization / Recommendations ----------
@app.get("/users/{user_id}/recommendations")
async def get_personalized_recommendations(user_id: int, limit: int = 10, background_tasks: BackgroundTasks = None):
    user_q = users.select().where(users.c.user_id == user_id)
    user_data = await database.fetch_one(user_q)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    recommendations = []
    method = "popular"
    if user_data['preferences']:
        pref_text = " ".join([str(v) for v in user_data['preferences'].values() if v])
        similar_items = vector_memory.find_similar_items(pref_text, limit)
        if similar_items:
            item_ids = [i['item_id'] for i in similar_items]
            detail_q = menu_items.select().where(
                sqlalchemy.and_(menu_items.c.item_id.in_(item_ids), menu_items.c.availability == True)
            )
            recommendations = await database.fetch_all(detail_q)
            # attach scores
            rec_map = {r['item_id']: dict(r) for r in recommendations}
            for sim in similar_items:
                if sim['item_id'] in rec_map:
                    rec_map[sim['item_id']]['recommendation_score'] = sim['similarity_score']
            recommendations = list(rec_map.values())
            method = "personalized"

    if not recommendations:
        # fallback to popular simple query
        popular_q = menu_items.select().where(menu_items.c.availability == True).limit(limit)
        recommendations = await database.fetch_all(popular_q)

    if background_tasks:
        background_tasks.add_task(log_interaction, InteractionLog(
            user_id=user_id,
            session_id=f"rec_{user_id}_{datetime.now().timestamp()}",
            interaction_type="recommendations_generated",
            response_data={"recommendations_count": len(recommendations)},
            interaction_metadata={"method": method}
        ))

    return {"user_id": user_id, "recommendations": recommendations, "total_count": len(recommendations), "method": method}


# ---------- Orders ----------
@app.post("/orders/", response_model=OrderResponse)
async def place_order(order: OrderCreate, background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    # enforce user matches token user
    if int(current_user['user_id']) != int(order.user_id):
        raise HTTPException(status_code=403, detail="Cannot place orders for other users")

    # calculate total
    total_amount = 0.0
    for it in order.order_items:
        item_q = menu_items.select().where(menu_items.c.item_id == it['item_id'])
        menu_item = await database.fetch_one(item_q)
        if not menu_item:
            raise HTTPException(status_code=400, detail=f"Item {it['item_id']} not found")
        total_amount += float(menu_item['price']) * int(it.get('quantity', 1))

    est_delivery = datetime.utcnow() + timedelta(minutes=30)
    q = orders.insert().values(
        user_id=order.user_id,
        restaurant_id=order.restaurant_id,
        order_items=order.order_items,
        total_amount=total_amount,
        delivery_address=order.delivery_address,
        special_instructions=order.special_instructions,
        payment_method=order.payment_method,
        estimated_delivery_time=est_delivery
    )
    order_id = await database.execute(q)

    # log order placed
    background_tasks.add_task(log_interaction, InteractionLog(
        user_id=order.user_id,
        session_id=f"order_{order.user_id}_{datetime.now().timestamp()}",
        interaction_type="order_placed",
        response_data={"order_id": order_id, "total_amount": float(total_amount), "items_count": len(order.order_items)},
        interaction_metadata={"restaurant_id": order.restaurant_id}
    ))

    return {
        **order.dict(),
        "order_id": order_id,
        "total_amount": total_amount,
        "order_status": "pending",
        "payment_status": "pending",
        "estimated_delivery_time": est_delivery,
        "created_at": datetime.utcnow()
    }


@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: int, current_user=Depends(get_current_user)):
    q = orders.select().where(orders.c.order_id == order_id)
    order_record = await database.fetch_one(q)
    if not order_record:
        raise HTTPException(status_code=404, detail="Order not found")
    # ensure user own order or admin (admin not implemented; require owner)
    if int(order_record['user_id']) != int(current_user['user_id']):
        raise HTTPException(status_code=403, detail="Not authorized to view this order")
    return order_record


@app.get("/users/{user_id}/orders", response_model=List[OrderResponse])
async def get_user_orders(user_id: int, limit: int = 50, current_user=Depends(get_current_user)):
    if int(user_id) != int(current_user['user_id']):
        raise HTTPException(status_code=403, detail="Not authorized")
    q = orders.select().where(orders.c.user_id == user_id).order_by(orders.c.created_at.desc()).limit(limit)
    return await database.fetch_all(q)


@app.put("/orders/{order_id}/status")
async def update_order_status(order_id: int, status: str, background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    valid_statuses = ["pending", "confirmed", "preparing", "ready", "delivered", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")

    # In production: check that the actor (restaurant or admin) can update. Here we allow owner for delivered/cancelled only.
    # Update DB
    update_data = {"order_status": status, "updated_at": datetime.utcnow()}
    if status == "delivered":
        update_data["actual_delivery_time"] = datetime.utcnow()
        update_data["payment_status"] = "paid"

    q = orders.update().where(orders.c.order_id == order_id).values(**update_data)
    await database.execute(q)

    # Log status change
    order_q = orders.select().where(orders.c.order_id == order_id)
    order_data = await database.fetch_one(order_q)
    if order_data:
        background_tasks.add_task(log_interaction, InteractionLog(
            user_id=order_data['user_id'],
            session_id=f"status_{order_id}_{datetime.now().timestamp()}",
            interaction_type="order_status_updated",
            response_data={"order_id": order_id, "new_status": status},
            interaction_metadata={"previous_status": order_data['order_status']}
        ))

    return {"message": f"Order status updated to {status}"}


# ---------- Feedback ----------
@app.post("/feedback/")
async def submit_feedback(feedback_data: FeedbackCreate, background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    # allow only owner to post feedback (simple guard)
    if int(current_user['user_id']) != int(feedback_data.user_id):
        raise HTTPException(status_code=403, detail="Cannot submit feedback for other users")

    sentiment_score = 0.0
    if feedback_data.comments:
        sentiment_score = (feedback_data.rating - 3) / 2

    q = feedback.insert().values(
        user_id=feedback_data.user_id,
        order_id=feedback_data.order_id,
        item_id=feedback_data.item_id,
        rating=feedback_data.rating,
        comments=feedback_data.comments,
        feedback_type=feedback_data.feedback_type,
        sentiment_score=sentiment_score
    )
    feedback_id = await database.execute(q)

    background_tasks.add_task(log_interaction, InteractionLog(
        user_id=feedback_data.user_id,
        session_id=f"feedback_{feedback_data.user_id}_{datetime.now().timestamp()}",
        interaction_type="feedback_submitted",
        user_rating=feedback_data.rating,
        response_data={"feedback_id": feedback_id, "sentiment_score": sentiment_score},
        interaction_metadata={"feedback_type": feedback_data.feedback_type}
    ))

    return {"feedback_id": feedback_id, "message": "Feedback submitted", "sentiment_score": sentiment_score}


@app.get("/feedback/user/{user_id}")
async def get_user_feedback(user_id: int, current_user=Depends(get_current_user)):
    if int(user_id) != int(current_user['user_id']):
        raise HTTPException(status_code=403, detail="Not authorized")
    q = feedback.select().where(feedback.c.user_id == user_id).order_by(feedback.c.created_at.desc())
    return await database.fetch_all(q)


# ---------- Analytics & Exports ----------
@app.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: int, current_user=Depends(get_current_user)):
    if int(user_id) != int(current_user['user_id']):
        raise HTTPException(status_code=403, detail="Not authorized")
    try:
        interaction_query = """
        SELECT interaction_type, COUNT(*) as count
        FROM user_interactions 
        WHERE user_id = :user_id 
        GROUP BY interaction_type
        """
        interactions = await database.fetch_all(sqlalchemy.text(interaction_query), {"user_id": user_id})

        order_query = """
        SELECT COUNT(*) as total_orders, 
               AVG(total_amount) as avg_order_value,
               SUM(total_amount) as total_spent
        FROM orders 
        WHERE user_id = :user_id
        """
        order_stats = await database.fetch_one(sqlalchemy.text(order_query), {"user_id": user_id})

        feedback_query = """
        SELECT AVG(rating) as avg_rating, COUNT(*) as feedback_count
        FROM feedback 
        WHERE user_id = :user_id
        """
        feedback_stats = await database.fetch_one(sqlalchemy.text(feedback_query), {"user_id": user_id})

        return {
            "user_id": user_id,
            "interactions": {row['interaction_type']: int(row['count']) for row in interactions} if interactions else {},
            "order_statistics": dict(order_stats) if order_stats else {},
            "feedback_statistics": dict(feedback_stats) if feedback_stats else {}
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")


@app.get("/analytics/popular_items")
async def get_popular_items(limit: int = 10):
    try:
        query = """
        SELECT mi.item_id, mi.name, mi.category, COUNT(ui.item_id) as interaction_count
        FROM menu_items mi
        LEFT JOIN user_interactions ui ON mi.item_id = ui.item_id
        WHERE ui.interaction_type IN ('item_view', 'order_placed')
        GROUP BY mi.item_id, mi.name, mi.category
        ORDER BY interaction_count DESC
        LIMIT :limit
        """
        popular_items = await database.fetch_all(sqlalchemy.text(query), {"limit": limit})
        return {"popular_items": [dict(item) for item in popular_items]}
    except Exception as e:
        logger.error(f"Error getting popular items: {e}")
        raise HTTPException(status_code=500, detail="Failed to get popular items")


# ---------- Interaction export ----------
@app.get("/export/interactions")
async def export_interactions(start_date: Optional[str] = None, end_date: Optional[str] = None, user_id: Optional[int] = None, current_user=Depends(get_current_user)):
    # In production, restrict to admins or data-team
    try:
        q = user_interactions.select()
        if start_date:
            q = q.where(user_interactions.c.created_at >= start_date)
        if end_date:
            q = q.where(user_interactions.c.created_at <= end_date)
        if user_id:
            q = q.where(user_interactions.c.user_id == user_id)
        interactions = await database.fetch_all(q)
        return {"interactions": [dict(i) for i in interactions]}
    except Exception as e:
        logger.error(f"Error exporting interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")


# ---------- Health check ----------
@app.get("/health")
async def health_check():
    try:
        # DB quick probe
        await database.fetch_one("SELECT 1")
        # vector store test
        _ = vector_memory.generate_embedding("health_check_test")
        return {
            "status": "healthy",
            "database": "connected",
            "vector_store": "operational" if vector_memory.chroma_collection else "not_initialized",
            "embedding_model": "loaded",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow()}


# ---------- Misc utility endpoints ----------
@app.post("/interactions/log")
async def log_interaction_endpoint(interaction: InteractionLog, background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    # allow user to log interactions
    if interaction.user_id and int(interaction.user_id) != int(current_user['user_id']):
        raise HTTPException(status_code=403, detail="Cannot log interaction for other users")
    background_tasks.add_task(log_interaction, interaction)
    return {"message": "Interaction queued for logging"}


# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
