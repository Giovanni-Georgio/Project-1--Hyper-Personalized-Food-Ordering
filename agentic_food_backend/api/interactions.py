# agentic_food_backend/api/interactions.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import List, Optional
from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import user_interactions
from agentic_food_backend.db.schemas import InteractionLog
from agentic_food_backend.api.dependencies import get_current_active_user
from agentic_food_backend.services.logging_service import log_interaction
from datetime import datetime
import sqlalchemy

router = APIRouter()

@router.post("/log")
async def log_interaction_endpoint(
    interaction: InteractionLog,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user)
):
    """Log user interaction"""
    # Verify user authorization
    if interaction.user_id and int(interaction.user_id) != int(current_user['user_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot log interaction for other users"
        )
    
    try:
        background_tasks.add_task(log_interaction, **interaction.dict())
        return {"message": "Interaction queued for logging"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log interaction"
        )

@router.get("/user/{user_id}")
async def get_user_interactions(
    user_id: int,
    interaction_type: Optional[str] = None,
    limit: int = 100,
    current_user = Depends(get_current_active_user)
):
    """Get user interaction history"""
    if int(user_id) != int(current_user['user_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view interactions for this user"
        )
    
    try:
        query = user_interactions.select().where(
            user_interactions.c.user_id == user_id
        ).order_by(user_interactions.c.created_at.desc()).limit(limit)
        
        if interaction_type:
            query = query.where(user_interactions.c.interaction_type == interaction_type)
        
        result = await database.fetch_all(query)
        return [dict(row) for row in result]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user interactions"
        )

@router.get("/analytics/user/{user_id}")
async def get_user_analytics(
    user_id: int,
    current_user = Depends(get_current_active_user)
):
    """Get user behavior analytics"""
    if int(user_id) != int(current_user['user_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view analytics for this user"
        )
    
    try:
        # Interaction summary
        interaction_query = """
        SELECT interaction_type, COUNT(*) as count
        FROM user_interactions 
        WHERE user_id = :user_id 
        GROUP BY interaction_type
        ORDER BY count DESC
        """
        interactions = await database.fetch_all(
            sqlalchemy.text(interaction_query), 
            {"user_id": user_id}
        )
        
        # Order summary
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
        
        # Feedback summary
        feedback_query = """
        SELECT AVG(rating) as avg_rating, COUNT(*) as feedback_count
        FROM feedback 
        WHERE user_id = :user_id
        """
        feedback_stats = await database.fetch_one(
            sqlalchemy.text(feedback_query), 
            {"user_id": user_id}
        )
        
        # Recent activity (last 30 days)
        activity_query = """
        SELECT DATE(created_at) as activity_date, 
               COUNT(*) as daily_interactions
        FROM user_interactions 
        WHERE user_id = :user_id 
        AND created_at >= NOW() - INTERVAL '30 days'
        GROUP BY DATE(created_at)
        ORDER BY activity_date DESC
        """
        recent_activity = await database.fetch_all(
            sqlalchemy.text(activity_query), 
            {"user_id": user_id}
        )
        
        return {
            "user_id": user_id,
            "interaction_summary": {row['interaction_type']: int(row['count']) for row in interactions},
            "order_statistics": dict(order_stats) if order_stats else {},
            "feedback_statistics": dict(feedback_stats) if feedback_stats else {},
            "recent_activity": [dict(row) for row in recent_activity]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user analytics"
        )

@router.get("/analytics/popular_items")
async def get_popular_items(limit: int = 10):
    """Get most popular menu items based on interactions"""
    try:
        query = """
        SELECT mi.item_id, mi.name, mi.category, 
               COUNT(ui.item_id) as interaction_count,
               AVG(CASE WHEN ui.user_rating IS NOT NULL THEN ui.user_rating END) as avg_rating
        FROM menu_items mi
        LEFT JOIN user_interactions ui ON mi.item_id = ui.item_id
        WHERE ui.interaction_type IN ('item_view', 'order_placed')
        GROUP BY mi.item_id, mi.name, mi.category
        HAVING COUNT(ui.item_id) > 0
        ORDER BY interaction_count DESC
        LIMIT :limit
        """
        popular_items = await database.fetch_all(
            sqlalchemy.text(query), 
            {"limit": limit}
        )
        
        return {"popular_items": [dict(item) for item in popular_items]}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get popular items"
        )

@router.get("/export/interactions")
async def export_interactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[int] = None,
    interaction_type: Optional[str] = None,
    current_user = Depends(get_current_active_user)
):
    """Export interaction data for analysis"""
    # In production, restrict to admin users or data team
    try:
        query = user_interactions.select()
        
        if start_date:
            query = query.where(user_interactions.c.created_at >= start_date)
        if end_date:
            query = query.where(user_interactions.c.created_at <= end_date)
        if user_id:
            query = query.where(user_interactions.c.user_id == user_id)
        if interaction_type:
            query = query.where(user_interactions.c.interaction_type == interaction_type)
        
        # Limit export size for performance
        query = query.limit(10000)
        
        interactions = await database.fetch_all(query)
        
        return {
            "interactions": [dict(interaction) for interaction in interactions],
            "total_count": len(interactions),
            "export_timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export interaction data"
        )

@router.get("/analytics/search_trends")
async def get_search_trends(
    days: int = 30,
    current_user = Depends(get_current_active_user)
):
    """Get search trends and popular queries"""
    try:
        # Top search queries
        query_trends = """
        SELECT query_text, COUNT(*) as search_count
        FROM user_interactions 
        WHERE interaction_type = 'menu_search' 
        AND query_text IS NOT NULL
        AND created_at >= NOW() - INTERVAL '{} days'
        GROUP BY query_text
        ORDER BY search_count DESC
        LIMIT 20
        """.format(days)
        
        top_queries = await database.fetch_all(sqlalchemy.text(query_trends))
        
        # Search volume over time
        volume_trends = """
        SELECT DATE(created_at) as search_date, COUNT(*) as daily_searches
        FROM user_interactions 
        WHERE interaction_type = 'menu_search'
        AND created_at >= NOW() - INTERVAL '{} days'
        GROUP BY DATE(created_at)
        ORDER BY search_date DESC
        """.format(days)
        
        search_volume = await database.fetch_all(sqlalchemy.text(volume_trends))
        
        return {
            "period_days": days,
            "top_queries": [dict(row) for row in top_queries],
            "search_volume": [dict(row) for row in search_volume]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search trends"
        )

@router.delete("/{interaction_id}")
async def delete_interaction(
    interaction_id: int,
    current_user = Depends(get_current_active_user)
):
    """Delete a specific interaction record"""
    # Check if interaction exists and belongs to user
    interaction_query = user_interactions.select().where(
        user_interactions.c.interaction_id == interaction_id
    )
    interaction = await database.fetch_one(interaction_query)
    
    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found"
        )
    
    # Verify user owns the interaction
    if interaction['user_id'] and int(interaction['user_id']) != int(current_user['user_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this interaction"
        )
    
    try:
        delete_query = user_interactions.delete().where(
            user_interactions.c.interaction_id == interaction_id
        )
        await database.execute(delete_query)
        
        return {"message": "Interaction deleted successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete interaction"
        )