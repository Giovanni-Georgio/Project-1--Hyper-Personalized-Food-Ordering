# agentic_food_backend/api/orders.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import List, Optional
from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import orders, menu_items, users
from agentic_food_backend.db.schemas import OrderCreate, OrderResponse
from agentic_food_backend.api.dependencies import get_current_active_user
from agentic_food_backend.services.logging_service import log_interaction
from datetime import datetime, timedelta
import sqlalchemy

router = APIRouter()

@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def place_order(
    order: OrderCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user)
):
    """Place a new order"""
    # Enforce user matches token user
    if int(current_user['user_id']) != int(order.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot place orders for other users"
        )
    
    try:
        # Calculate total amount
        total_amount = 0.0
        for item in order.order_items:
            item_query = menu_items.select().where(menu_items.c.item_id == item['item_id'])
            menu_item = await database.fetch_one(item_query)
            
            if not menu_item:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Menu item {item['item_id']} not found"
                )
            
            if not menu_item['availability']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Menu item {menu_item['name']} is not available"
                )
            
            quantity = item.get('quantity', 1)
            total_amount += float(menu_item['price']) * quantity
        
        # Estimate delivery time
        estimated_delivery = datetime.utcnow() + timedelta(minutes=30)
        
        # Create order
        query = orders.insert().values(
            user_id=order.user_id,
            restaurant_id=order.restaurant_id,
            order_items=order.order_items,
            total_amount=total_amount,
            delivery_address=order.delivery_address,
            special_instructions=order.special_instructions,
            payment_method=order.payment_method,
            estimated_delivery_time=estimated_delivery
        )
        order_id = await database.execute(query)
        
        # Log order interaction
        background_tasks.add_task(
            log_interaction,
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
        
        return {
            **order.dict(),
            "order_id": order_id,
            "total_amount": total_amount,
            "order_status": "pending",
            "payment_status": "pending",
            "estimated_delivery_time": estimated_delivery,
            "created_at": datetime.utcnow()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to place order"
        )

@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: int, current_user = Depends(get_current_active_user)):
    """Get order by ID"""
    query = orders.select().where(orders.c.order_id == order_id)
    order = await database.fetch_one(query)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    # Ensure user owns the order
    if int(order['user_id']) != int(current_user['user_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this order"
        )
    
    return order

@router.get("/user/{user_id}", response_model=List[OrderResponse])
async def get_user_orders(
    user_id: int,
    limit: int = 50,
    current_user = Depends(get_current_active_user)
):
    """Get user's order history"""
    if int(user_id) != int(current_user['user_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view orders for this user"
        )
    
    try:
        query = orders.select().where(orders.c.user_id == user_id).order_by(
            orders.c.created_at.desc()
        ).limit(limit)
        
        result = await database.fetch_all(query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user orders"
        )

@router.put("/{order_id}/status")
async def update_order_status(
    order_id: int,
    status_update: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user)
):
    """Update order status"""
    valid_statuses = ["pending", "confirmed", "preparing", "ready", "delivered", "cancelled"]
    
    if status_update not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )
    
    # Check if order exists
    order_query = orders.select().where(orders.c.order_id == order_id)
    order_data = await database.fetch_one(order_query)
    
    if not order_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    try:
        # Update order status
        update_data = {
            "order_status": status_update,
            "updated_at": datetime.utcnow()
        }
        
        if status_update == "delivered":
            update_data["actual_delivery_time"] = datetime.utcnow()
            update_data["payment_status"] = "paid"
        
        query = orders.update().where(orders.c.order_id == order_id).values(**update_data)
        await database.execute(query)
        
        # Log status update
        background_tasks.add_task(
            log_interaction,
            user_id=order_data['user_id'],
            session_id=f"status_{order_id}_{datetime.now().timestamp()}",
            interaction_type="order_status_updated",
            response_data={
                "order_id": order_id,
                "new_status": status_update,
                "previous_status": order_data['order_status']
            }
        )
        
        return {"message": f"Order status updated to {status_update}"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update order status"
        )

@router.get("/", response_model=List[OrderResponse])
async def list_orders(
    status_filter: Optional[str] = None,
    limit: int = 100,
    current_user = Depends(get_current_active_user)
):
    """List orders (for admin or restaurant staff)"""
    # In production, add proper role-based access control
    try:
        query = orders.select().order_by(orders.c.created_at.desc()).limit(limit)
        
        if status_filter:
            query = query.where(orders.c.order_status == status_filter)
        
        result = await database.fetch_all(query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch orders"
        )

@router.delete("/{order_id}")
async def cancel_order(
    order_id: int,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user)
):
    """Cancel an order"""
    # Check if order exists
    order_query = orders.select().where(orders.c.order_id == order_id)
    order_data = await database.fetch_one(order_query)
    
    if not order_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    # Ensure user owns the order
    if int(order_data['user_id']) != int(current_user['user_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to cancel this order"
        )
    
    # Check if order can be cancelled
    if order_data['order_status'] in ["delivered", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel order with status: {order_data['order_status']}"
        )
    
    try:
        # Update order status to cancelled
        query = orders.update().where(orders.c.order_id == order_id).values(
            order_status="cancelled",
            updated_at=datetime.utcnow()
        )
        await database.execute(query)
        
        # Log cancellation
        background_tasks.add_task(
            log_interaction,
            user_id=order_data['user_id'],
            session_id=f"cancel_{order_id}_{datetime.now().timestamp()}",
            interaction_type="order_cancelled",
            response_data={"order_id": order_id}
        )
        
        return {"message": "Order cancelled successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel order"
        )