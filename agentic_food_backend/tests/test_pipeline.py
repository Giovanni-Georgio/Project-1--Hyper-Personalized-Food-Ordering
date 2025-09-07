import pytest
from httpx import AsyncClient
from agentic_food_backend.main import app
from agentic_food_backend.db.database import database
from agentic_food_backend.db.models import restaurants, menu_items

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as c:
        await database.connect()
        # Create a test restaurant
        await database.execute(restaurants.insert().values(name="Test", platform="generic", url="http://x"))
        yield c
        await database.execute(menu_items.delete())  # cleanup
        await database.disconnect()

@pytest.mark.anyio
async def test_ocr_ingest_and_db(client):
    # 1. Simulate OCR extract
    sample_menu_data = {
        "sections": [{"name":"Menu","items":[{"name":"Cake","description":"Delicious","price":5.0,"currency":"USD","availability":True}]}]
    }
    # 2. Call ingest endpoint
    resp = await client.post(
        "/api/v1/ocr/ingest",
        params={"restaurant_id": 1},
        json=sample_menu_data
    )
    assert resp.status_code == 201
    
    # Wait briefly for background task
    await asyncio.sleep(0.1)
    
    # 3. Verify DB record
    rows = await database.fetch_all(menu_items.select().where(menu_items.c.restaurant_id == 1))
    assert len(rows) == 1
    assert rows[0]["name"] == "Cake"
