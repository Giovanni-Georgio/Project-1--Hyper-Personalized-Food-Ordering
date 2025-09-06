# api_endpoints.py

from fastapi import FastAPI, HTTPException, Query
from scraper.menu_scraper import scrape_and_ocr_menu

app = FastAPI()

@app.get("/menu")
async def get_menu(url: str = Query(..., description="URL of the restaurant menu page")):
    try:
        menu_data = scrape_and_ocr_menu(url)
        return {"menu": menu_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
