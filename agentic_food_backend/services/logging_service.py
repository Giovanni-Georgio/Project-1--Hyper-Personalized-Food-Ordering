# agentic_food_backend/services/logging_service.py

import logging
from datetime import datetime
from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import user_interactions

logger = logging.getLogger(__name__)

async def log_interaction(
    user_id: int = None,
    session_id: str = None,
    interaction_type: str = None,
    item_id: int = None,
    query_text: str = None,
    response_data: dict = None,
    user_rating: int = None,
    interaction_metadata: dict = None
):
    """Log user interaction to database"""
    try:
        query = user_interactions.insert().values(
            user_id=user_id,
            session_id=session_id,
            interaction_type=interaction_type,
            item_id=item_id,
            query_text=query_text,
            response_data=response_data,
            user_rating=user_rating,
            interaction_metadata=interaction_metadata,
            created_at=datetime.utcnow()
        )
        await database.execute(query)
        logger.info(f"Logged interaction: {interaction_type} for user {user_id}")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

class InteractionLogger:
    """Service for logging user interactions with structured data"""
    
    @staticmethod
    async def log_search(user_id: int, query: str, results_count: int, method: str = "text"):
        """Log search interaction"""
        await log_interaction(
            user_id=user_id,
            session_id=f"search_{user_id}_{datetime.now().timestamp()}",
            interaction_type="menu_search",
            query_text=query,
            response_data={"results_count": results_count},
            interaction_metadata={"search_method": method}
        )
    
    @staticmethod
    async def log_item_view(user_id: int, item_id: int, item_name: str):
        """Log menu item view"""
        await log_interaction(
            user_id=user_id,
            session_id=f"view_{user_id}_{datetime.now().timestamp()}",
            interaction_type="item_view",
            item_id=item_id,
            interaction_metadata={"item_name": item_name}
        )
    
    @staticmethod
    async def log_order_placed(user_id: int, order_id: int, total_amount: float, items_count: int, restaurant_id: int):
        """Log order placement"""
        await log_interaction(
            user_id=user_id,
            session_id=f"order_{user_id}_{datetime.now().timestamp()}",
            interaction_type="order_placed",
            response_data={
                "order_id": order_id,
                "total_amount": total_amount,
                "items_count": items_count
            },
            interaction_metadata={"restaurant_id": restaurant_id}
        )
    
    @staticmethod
    async def log_recommendations_generated(user_id: int, recommendations_count: int, method: str):
        """Log recommendation generation"""
        await log_interaction(
            user_id=user_id,
            session_id=f"rec_{user_id}_{datetime.now().timestamp()}",
            interaction_type="recommendations_generated",
            response_data={"recommendations_count": recommendations_count},
            interaction_metadata={"method": method}
        )
    
    @staticmethod
    async def log_feedback_submitted(user_id: int, feedback_id: int, rating: int, feedback_type: str, sentiment_score: float):
        """Log feedback submission"""
        await log_interaction(
            user_id=user_id,
            session_id=f"feedback_{user_id}_{datetime.now().timestamp()}",
            interaction_type="feedback_submitted",
            user_rating=rating,
            response_data={
                "feedback_id": feedback_id,
                "sentiment_score": sentiment_score
            },
            interaction_metadata={"feedback_type": feedback_type}
        )
    
    @staticmethod
    async def log_user_registration(user_id: int, email: str):
        """Log user registration"""
        await log_interaction(
            user_id=user_id,
            session_id=f"reg_{user_id}",
            interaction_type="user_registration",
            interaction_metadata={"email": email}
        )
    
    @staticmethod
    async def log_order_status_update(user_id: int, order_id: int, new_status: str, previous_status: str = None):
        """Log order status update"""
        await log_interaction(
            user_id=user_id,
            session_id=f"status_{order_id}_{datetime.now().timestamp()}",
            interaction_type="order_status_updated",
            response_data={
                "order_id": order_id,
                "new_status": new_status,
                "previous_status": previous_status
            }
        )
    
    @staticmethod
    async def log_custom_interaction(
        user_id: int,
        interaction_type: str,
        session_id: str = None,
        item_id: int = None,
        query_text: str = None,
        response_data: dict = None,
        user_rating: int = None,
        interaction_metadata: dict = None
    ):
        """Log custom interaction with flexible parameters"""
        if not session_id:
            session_id = f"{interaction_type}_{user_id}_{datetime.now().timestamp()}"
        
        await log_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_type=interaction_type,
            item_id=item_id,
            query_text=query_text,
            response_data=response_data,
            user_rating=user_rating,
            interaction_metadata=interaction_metadata
        )

# Global logger instance
interaction_logger = InteractionLogger()