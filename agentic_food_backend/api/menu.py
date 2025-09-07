# agentic_food_backend/api/menu.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import List, Optional
from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import menu_items, restaurants
from agentic_food_backend.db.schemas import MenuItemCreate, MenuItemResponse
from agentic_food_backend.api.dependencies import get_current_active_user, get_current_user
from agentic_food_backend.services.embeddings import vector_memory
from agentic_food_backend.services.logging_service import log_interaction
from datetime import datetime
import sqlalchemy

router = APIRouter()

@router.post("/", response_model=MenuItemResponse, status_code=status.HTTP_201_CREATED)
async def create_menu_item(
    item: MenuItemCreate, 
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user)
):
    """Create a new menu item"""
    # Verify restaurant exists
    restaurant = await database.fetch_one(
        restaurants.select().where(restaurants.c.restaurant_id == item.restaurant_id)
    )
    
    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found"
        )
    
    try:
        query = menu_items.insert().values(**item.dict())
        item_id = await database.execute(query)
        
        # Add to vector store in background
        background_tasks.add_task(
            vector_memory.add_menu_item_embedding,
            item_id,
            item.name,
            item.description or "",
            item.category or ""
        )
        
        return {
            **item.dict(),
            "item_id": item_id,
            "availability": True,
            "created_at": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create menu item"
        )

@router.get("/", response_model=List[MenuItemResponse])
async def list_menu_items(
    restaurant_id: Optional[int] = None,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    available_only: bool = True,
    limit: int = 100
):
    """List menu items with filters"""
    try:
        query = menu_items.select().limit(limit)
        
        if restaurant_id:
            query = query.where(menu_items.c.restaurant_id == restaurant_id)
        if category:
            query = query.where(menu_items.c.category.ilike(f"%{category}%"))
        if max_price:
            query = query.where(menu_items.c.price <= max_price)
        if available_only:
            query = query.where(menu_items.c.availability == True)
        
        result = await database.fetch_all(query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch menu items"
        )

@router.get("/search")
async def search_menu_items(
    query: str,
    user_id: Optional[int] = None,
    limit: int = 20,
    background_tasks: BackgroundTasks = None
):
    """Intelligent menu item search using vector similarity"""
    try:
        # Try vector similarity search first
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
            method = "text"
        else:
            # Get full item details for similar items
            item_ids = [item['item_id'] for item in similar_items]
            detail_query = menu_items.select().where(menu_items.c.item_id.in_(item_ids))
            results = await database.fetch_all(detail_query)
            
            # Attach similarity scores
            results_map = {r['item_id']: dict(r) for r in results}
            for sim in similar_items:
                if sim['item_id'] in results_map:
                    results_map[sim['item_id']]['similarity_score'] = sim['similarity_score']
            results = list(results_map.values())
            method = "vector"
        
        # Log search interaction
        if background_tasks and user_id:
            background_tasks.add_task(
                log_interaction,
                user_id=user_id,
                session_id=f"search_{user_id}_{datetime.now().timestamp()}",
                interaction_type="menu_search",
                query_text=query,
                response_data={"results_count": len(results)},
                interaction_metadata={"search_method": method}
            )
        
        return {
            "query": query,
            "results": results,
            "total_count": len(results),
            "method": method
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )

@router.get("/{item_id}", response_model=MenuItemResponse)
async def get_menu_item(
    item_id: int, 
    user_id: Optional[int] = None, 
    background_tasks: BackgroundTasks = None
):
    """Get menu item by ID"""
    query = menu_items.select().where(menu_items.c.item_id == item_id)
    item = await database.fetch_one(query)
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Menu item not found"
        )
    
    # Log view interaction if user present
    if background_tasks and user_id:
        background_tasks.add_task(
            log_interaction,
            user_id=user_id,
            session_id=f"view_{user_id}_{datetime.now().timestamp()}",
            interaction_type="item_view",
            item_id=item_id,
            interaction_metadata={"item_name": item['name']}
        )
    
    return item

@router.put("/{item_id}", response_model=MenuItemResponse)
async def update_menu_item(
    item_id: int,
    item_update: MenuItemCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user)
):
    """Update menu item"""
    # Check if item exists
    existing = await database.fetch_one(
        menu_items.select().where(menu_items.c.item_id == item_id)
    )
    
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Menu item not found"
        )
    
    try:
        query = menu_items.update().where(menu_items.c.item_id == item_id).values(
            **item_update.dict(),
            updated_at=datetime.utcnow()
        )
        await database.execute(query)
        
        # Update vector embedding in background
        background_tasks.add_task(
            vector_memory.add_menu_item_embedding,
            item_id,
            item_update.name,
            item_update.description or "",
            item_update.category or ""
        )
        
        # Return updated item
        updated_item = await database.fetch_one(
            menu_items.select().where(menu_items.c.item_id == item_id)
        )
        return updated_item
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update menu item"
        )

@router.delete("/{item_id}")
async def delete_menu_item(
    item_id: int,
    current_user = Depends(get_current_active_user)
):
    """Soft delete menu item"""
    # Check if item exists
    existing = await database.fetch_one(
        menu_items.select().where(menu_items.c.item_id == item_id)
    )
    
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Menu item not found"
        )
    
    try:
        query = menu_items.update().where(menu_items.c.item_id == item_id).values(
            availability=False,
            updated_at=datetime.utcnow()
        )
        await database.execute(query)
        
        return {"message": "Menu item deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete menu item"
        )

@router.get("/recommendations/{user_id}")
async def get_personalized_recommendations(
    user_id: int,
    limit: int = 10,
    background_tasks: BackgroundTasks = None,
    current_user = Depends(get_current_user)
):
    """Get personalized menu recommendations for user"""
    # Verify user authorization
    if int(current_user['user_id']) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view recommendations for this user"
        )
    
    try:
        from agentic_food_backend.db.models import users
        
        # Get user preferences
        user_query = users.select().where(users.c.user_id == user_id)
        user_data = await database.fetch_one(user_query)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        recommendations = []
        method = "popular"
        
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
                rec_map = {r['item_id']: dict(r) for r in recommendations}
                for sim in similar_items:
                    if sim['item_id'] in rec_map:
                        rec_map[sim['item_id']]['recommendation_score'] = sim['similarity_score']
                recommendations = list(rec_map.values())
                method = "personalized"
        
        # Fallback to popular items if no personalized recommendations
        if not recommendations:
            popular_query = menu_items.select().where(
                menu_items.c.availability == True
            ).limit(limit)
            recommendations = await database.fetch_all(popular_query)
        
        # Log recommendation interaction
        if background_tasks:
            background_tasks.add_task(
                log_interaction,
                user_id=user_id,
                session_id=f"rec_{user_id}_{datetime.now().timestamp()}",
                interaction_type="recommendations_generated",
                response_data={"recommendations_count": len(recommendations)},
                interaction_metadata={"method": method}
            )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "method": method
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )