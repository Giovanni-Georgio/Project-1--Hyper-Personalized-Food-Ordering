from databases import Database
from sqlalchemy import MetaData, create_engine
from ..config import DATABASE_URL, SYNC_DATABASE_URL #agentic_food_backend/config.py

database = Database(DATABASE_URL)
metadata = MetaData()
engine = create_engine(SYNC_DATABASE_URL)

async def connect_db():
    await database.connect()

async def disconnect_db():
    await database.disconnect()
