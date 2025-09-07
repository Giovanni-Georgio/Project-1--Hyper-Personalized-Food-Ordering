# Updated main.py with OCR and Scraper integration
# agentic_food_backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Database imports
from agentic_food_backend.db.database import database, connect_db, disconnect_db

# API router imports
from agentic_food_backend.api import (
    users, restaurants, menu, orders, feedback, interactions,
    ocr, scraper  # New OCR and scraper endpoints
)

# Service imports
from agentic_food_backend.services.chromadb_service import initialize_chroma_collection
from agentic_food_backend.services.faiss_service import faiss_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agentic AI Food Ordering Backend",
    description="Comprehensive backend for AI-powered food ordering with OCR and web scraping capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(restaurants.router, prefix="/api/v1/restaurants", tags=["Restaurants"])
app.include_router(menu.router, prefix="/api/v1/menu", tags=["Menu Items"])
app.include_router(orders.router, prefix="/api/v1/orders", tags=["Orders"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(interactions.router, prefix="/api/v1/interactions", tags=["Interactions"])

# NEW: Include OCR and Scraper routers
app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["OCR Processing"])
app.include_router(scraper.router, prefix="/api/v1/scraper", tags=["Web Scraping"])

from agentic_food_backend.api import debug_ocr
app.include_router(debug_ocr.router, prefix="/api/v1/debug", tags=["Debug OCR"])


@app.on_event("startup")
async def on_startup():
    """Initialize services on startup."""
    try:
        logger.info("Starting Agentic AI Food Ordering Backend...")
        
        # Connect to database
        await connect_db()
        logger.info("Database connection established")
        
        # Initialize ChromaDB collection
        await initialize_chroma_collection()
        logger.info("ChromaDB collection initialized")
        
        # Initialize FAISS service
        faiss_service.initialize_index()
        logger.info("FAISS index initialized")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on shutdown."""
    try:
        logger.info("Shutting down services...")
        
        # Disconnect from database
        await disconnect_db()
        logger.info("Database connection closed")
        
        # Save FAISS index
        faiss_service.save_index()
        logger.info("FAISS index saved")
        
        logger.info("Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Agentic AI Food Ordering Backend",
        "status": "running",
        "version": "1.0.0",
        "features": [
            "User Management",
            "Restaurant & Menu Management", 
            "Order Processing",
            "Feedback System",
            "User Interaction Analytics",
            "OCR Menu Processing",
            "Web Scraping",
            "Vector-based Recommendations"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check database connection
        db_status = "healthy"
        try:
            await database.fetch_val("SELECT 1")
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Check vector services
        from agentic_food_backend.services.embeddings import vector_memory
        vector_health = vector_memory.health_check()
        
        # Check OCR service
        from agentic_food_backend.services.ocr_service import ocr_service
        try:
            # Simple OCR test
            test_result = await ocr_service.process_image(
                image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                preprocessing=False
            )
            ocr_status = "healthy" if test_result is not None else "unhealthy"
        except Exception as e:
            ocr_status = f"unhealthy: {str(e)}"
        
        return {
            "status": "healthy",
            "timestamp": "2025-09-07T12:00:00Z",
            "services": {
                "database": db_status,
                "vector_memory": vector_health,
                "ocr_service": ocr_status,
                "api": "healthy"
            },
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-09-07T12:00:00Z"
        }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Global error: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "path": str(request.url)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agentic_food_backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

#from fastapi import FastAPI
#from fastapi.middleware.cors import CORSMiddleware

#from agentic_food_backend.db.database import database, connect_db, disconnect_db
#from agentic_food_backend.api import (
#    users, restaurants, menu, orders, feedback, interactions
#)
#from agentic_food_backend.services.chromadb_service import chroma_client, initialize_chroma_collection

#app = FastAPI(title="Agentic AI Food Ordering Backend")

#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)

#app.include_router(users.router, prefix="/users")
#app.include_router(restaurants.router, prefix="/restaurants")
#app.include_router(menu.router, prefix="/menu_items")
#app.include_router(orders.router, prefix="/orders")
#app.include_router(feedback.router, prefix="/feedback")
#app.include_router(interactions.router, prefix="/interactions")


#@app.on_event("startup")
#async def on_startup():
#    await connect_db()
#    await initialize_chroma_collection()  # Prepare vector DB


#@app.on_event("shutdown")
#async def on_shutdown():
#    await disconnect_db()

