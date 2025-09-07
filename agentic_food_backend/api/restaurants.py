# agentic_food_backend/api/restaurants.py

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import restaurants
from agentic_food_backend.db.schemas import RestaurantCreate, RestaurantResponse
from agentic_food_backend.api.dependencies import get_current_active_user
from datetime import datetime
import sqlalchemy

router = APIRouter()

@router.post("/", response_model=RestaurantResponse, status_code=status.HTTP_201_CREATED)
async def create_restaurant(
    restaurant: RestaurantCreate, 
    current_user = Depends(get_current_active_user)
):
    """Create a new restaurant"""
    try:
        query = restaurants.insert().values(**restaurant.dict())
        restaurant_id = await database.execute(query)
        
        return {
            **restaurant.dict(),
            "restaurant_id": restaurant_id,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create restaurant"
        )

@router.get("/", response_model=List[RestaurantResponse])
async def list_restaurants(
    cuisine_type: Optional[str] = None,
    location: Optional[str] = None,
    limit: int = 50
):
    """List all restaurants with optional filters"""
    try:
        query = restaurants.select().where(restaurants.c.is_active == True).limit(limit)
        
        if cuisine_type:
            query = query.where(restaurants.c.cuisine_type.ilike(f"%{cuisine_type}%"))
        if location:
            query = query.where(restaurants.c.location.ilike(f"%{location}%"))
        
        result = await database.fetch_all(query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch restaurants"
        )

@router.get("/{restaurant_id}", response_model=RestaurantResponse)
async def get_restaurant(restaurant_id: int):
    """Get restaurant by ID"""
    query = restaurants.select().where(restaurants.c.restaurant_id == restaurant_id)
    restaurant = await database.fetch_one(query)
    
    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found"
        )
    
    return restaurant

@router.put("/{restaurant_id}", response_model=RestaurantResponse)
async def update_restaurant(
    restaurant_id: int,
    restaurant_update: RestaurantCreate,
    current_user = Depends(get_current_active_user)
):
    """Update restaurant information"""
    # Check if restaurant exists
    existing = await database.fetch_one(
        restaurants.select().where(restaurants.c.restaurant_id == restaurant_id)
    )
    
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found"
        )
    
    try:
        query = restaurants.update().where(restaurants.c.restaurant_id == restaurant_id).values(
            **restaurant_update.dict(),
            updated_at=datetime.utcnow()
        )
        await database.execute(query)
        
        # Return updated restaurant
        updated_restaurant = await database.fetch_one(
            restaurants.select().where(restaurants.c.restaurant_id == restaurant_id)
        )
        return updated_restaurant
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update restaurant"
        )

@router.delete("/{restaurant_id}")
async def delete_restaurant(
    restaurant_id: int,
    current_user = Depends(get_current_active_user)
):
    """Soft delete restaurant"""
    # Check if restaurant exists
    existing = await database.fetch_one(
        restaurants.select().where(restaurants.c.restaurant_id == restaurant_id)
    )
    
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found"
        )
    
    try:
        query = restaurants.update().where(restaurants.c.restaurant_id == restaurant_id).values(
            is_active=False,
            updated_at=datetime.utcnow()
        )
        await database.execute(query)
        
        return {"message": "Restaurant deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete restaurant"
        )