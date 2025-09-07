from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from typing import List
from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import users
from agentic_food_backend.db.schemas import UserCreate, UserResponse
from agentic_food_backend.services.embeddings import vector_memory
from agentic_food_backend.api.dependencies import get_current_user
from datetime import datetime
from agentic_food_backend.db.schemas import InteractionLog  # Assuming interaction schema

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, background_tasks: BackgroundTasks):
    existing = await database.fetch_one(users.select().where(users.c.email == user.email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password omitted for brevity; must hash here

    query = users.insert().values(
        name=user.name,
        email=user.email,
        phone=user.phone,
        preferences=user.preferences,
        dietary_restrictions=user.dietary_restrictions,
        address=user.address,
        password_hash="hashed_password_here",  # Placeholder
        created_at=datetime.utcnow()
    )
    user_id = await database.execute(query)

    if user.preferences:
        background_tasks.add_task(vector_memory.add_user_preference_embedding, user_id, user.preferences)

    # Log registration interaction code omitted

    return {
        "user_id": user_id,
        **user.dict(exclude={"password"}),
        "created_at": datetime.utcnow(),
        "is_active": True,
    }

# Additional user endpoints here...
