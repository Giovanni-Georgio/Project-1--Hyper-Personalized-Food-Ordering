"""
Configuration settings for the Hyper-Personalized Food Ordering AI
"""

import os
from typing import Dict, List

class Config:
    """Application configuration"""
    
    # Hugging Face Settings
    HUGGINGFACE_API_TOKEN = "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"
    
    # Model Configuration
    PREFERRED_MODELS = [
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill", 
        "google/flan-t5-base",
        "microsoft/DialoGPT-small"
    ]
    
    MODEL_KWARGS = {
        "temperature": 0.7,
        "max_length": 200,
        "do_sample": True,
        "pad_token_id": 50256
    }
    
    # API Endpoints (Team Integration)
    MENU_API_BASE = "http://localhost:8001"
    BACKEND_API_BASE = "http://localhost:8002"
    
    # API Endpoints for Production (when deployed)
    MENU_API_BASE_PROD = "https://your-menu-api.herokuapp.com"
    BACKEND_API_BASE_PROD = "https://your-backend-api.herokuapp.com"
    
    # Application Settings
    APP_TITLE = "🍕 Hyper-Personalized Food Ordering AI"
    APP_ICON = "🍕"
    DEFAULT_USER_ID = "default_user"
    
    # Budget Settings
    MIN_BUDGET = 5
    MAX_BUDGET = 500
    DEFAULT_MIN_BUDGET = 10
    DEFAULT_MAX_BUDGET = 30
    
    # Dietary Restrictions Options
    DIETARY_OPTIONS = [
        "vegetarian", "vegan", "gluten-free", "dairy-free",
        "halal", "kosher", "keto", "paleo", "low-carb", "nut-free"
    ]
    
    # Cuisine Options
    CUISINE_OPTIONS = [
        "Italian", "Chinese", "Indian", "Mexican", "Thai", "Japanese",
        "Mediterranean", "American", "French", "Korean", "Vietnamese",
        "Greek", "Lebanese", "Spanish", "Healthy", "Fast Food"
    ]
    
    # Meal Time Options
    MEAL_TIME_OPTIONS = [
        "Breakfast", "Brunch", "Lunch", "Dinner", "Late Night", "Snacks"
    ]
    
    # Chat Settings
    MAX_CHAT_HISTORY = 50
    CHAT_INPUT_PLACEHOLDER = "💬 Ask me anything about food... (e.g., 'I want healthy Italian under $20')"
    
    # System Messages
    WELCOME_MESSAGE = """Welcome! 👋 

I'm your hyper-personalized AI food assistant. I've learned about your preferences and I'm ready to help you discover amazing meals that perfectly match your taste, budget, and dietary needs.

**What would you like to do today?**
- 🔍 Find restaurants and dishes
- 🛒 Place an order  
- 💡 Get personalized recommendations
- 📊 Explore new cuisines

Just start chatting naturally - I understand context and learn from our conversation!"""

    ERROR_MESSAGE = """I apologize, but I encountered an error while processing your request.

🔧 **Troubleshooting suggestions:**
- Try rephrasing your question
- Check if the AI model is properly loaded
- Verify your internet connection

Please try asking again, and I'll do my best to help you find great food options!"""
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Paths
    MEMORY_FILE_PREFIX = "user_memory_"
    MEMORY_FILE_EXTENSION = ".json"
    
    @classmethod
    def get_memory_file_path(cls, user_id: str) -> str:
        """Get the memory file path for a user"""
        return f"{cls.MEMORY_FILE_PREFIX}{user_id}{cls.MEMORY_FILE_EXTENSION}"
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return os.getenv("ENVIRONMENT", "development") == "development"
    
    @classmethod
    def get_api_base_url(cls, api_type: str) -> str:
        """Get the appropriate API base URL based on environment"""
        if cls.is_development():
            return cls.MENU_API_BASE if api_type == "menu" else cls.BACKEND_API_BASE
        else:
            return cls.MENU_API_BASE_PROD if api_type == "menu" else cls.BACKEND_API_BASE_PROD

# Quick Actions Configuration
QUICK_ACTIONS = [
    {
        "text": "🍕 Italian Favorites",
        "query": "Show me the best Italian dishes within my budget",
        "icon": "🍕"
    },
    {
        "text": "🥗 Healthy Options", 
        "query": "I want healthy, nutritious meals",
        "icon": "🥗"
    },
    {
        "text": "💰 Budget Meals",
        "query": "Show me the best value meals",
        "icon": "💰"
    },
    {
        "text": "🌶️ Spicy Food",
        "query": "I'm craving something spicy",
        "icon": "🌶️"
    },
    {
        "text": "🥘 Surprise Me",
        "query": "Recommend something new based on my taste profile",
        "icon": "🥘"
    },
    {
        "text": "⚡ Quick Delivery",
        "query": "What can I get delivered fastest?",
        "icon": "⚡"
    }
]

# Sample Restaurant Data (Fallback)
SAMPLE_RESTAURANTS = [
    {
        "id": "rest_001",
        "name": "Mario's Italian Bistro",
        "cuisine": "Italian",
        "rating": 4.5,
        "delivery_time": "25-35 min",
        "image_url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400",
        "items": [
            {
                "id": "item_001",
                "name": "Margherita Pizza",
                "price": 15.99,
                "description": "Fresh mozzarella, basil, tomato sauce on thin crust",
                "vegetarian": True,
                "vegan": False,
                "gluten_free": False,
                "calories": 280,
                "prep_time": "15-20 min",
                "ingredients": ["mozzarella", "basil", "tomato", "flour", "olive oil"],
                "image_url": "https://images.unsplash.com/photo-1604382355076-af4b0eb60143?w=300"
            },
            {
                "id": "item_002",
                "name": "Chicken Parmigiana",
                "price": 18.99,
                "description": "Breaded chicken breast with marinara and melted cheese",
                "vegetarian": False,
                "vegan": False,
                "gluten_free": False,
                "calories": 450,
                "prep_time": "20-25 min",
                "ingredients": ["chicken", "breadcrumbs", "marinara", "mozzarella"],
                "image_url": "https://images.unsplash.com/photo-1632778149955-e80f8ceca2e8?w=300"
            }
        ]
    },
    {
        "id": "rest_002", 
        "name": "Spice Garden Indian",
        "cuisine": "Indian",
        "rating": 4.3,
        "delivery_time": "30-40 min",
        "image_url": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=400",
        "items": [
            {
                "id": "item_003",
                "name": "Vegetable Biryani",
                "price": 14.99,
                "description": "Fragrant basmati rice with mixed vegetables and aromatic spices",
                "vegetarian": True,
                "vegan": True,
                "gluten_free": True,
                "calories": 320,
                "prep_time": "25-30 min",
                "ingredients": ["basmati rice", "mixed vegetables", "spices", "saffron"],
                "image_url": "https://images.unsplash.com/photo-1563379091339-03246963d51a?w=300"
            },
            {
                "id": "item_004",
                "name": "Butter Chicken",
                "price": 16.99,
                "description": "Tender chicken in rich, creamy tomato curry sauce",
                "vegetarian": False,
                "vegan": False,
                "gluten_free": True,
                "calories": 380,
                "prep_time": "20-25 min",
                "ingredients": ["chicken", "tomato", "cream", "butter", "spices"],
                "image_url": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=300"
            }
        ]
    },
    {
        "id": "rest_003",
        "name": "Green Bowl Healthy",
        "cuisine": "Healthy",
        "rating": 4.6,
        "delivery_time": "20-30 min",
        "image_url": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400",
        "items": [
            {
                "id": "item_005",
                "name": "Buddha Bowl",
                "price": 13.99,
                "description": "Quinoa base with roasted vegetables, avocado, and tahini dressing",
                "vegetarian": True,
                "vegan": True,
                "gluten_free": True,
                "calories": 420,
                "prep_time": "15-20 min",
                "ingredients": ["quinoa", "sweet potato", "broccoli", "avocado", "tahini"],
                "image_url": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=300"
            },
            {
                "id": "item_006",
                "name": "Grilled Salmon Bowl",
                "price": 19.99,
                "description": "Grilled salmon with quinoa, steamed vegetables, and lemon herb sauce",
                "vegetarian": False,
                "vegan": False,
                "gluten_free": True,
                "calories": 480,
                "prep_time": "18-22 min",
                "ingredients": ["salmon", "quinoa", "broccoli", "carrots", "lemon"],
                "image_url": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=300"
            }
        ]
    }
]