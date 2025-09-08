import json
import os
from typing import Dict, List, Any
from datetime import datetime

class UserMemory:
    """Handles user preference storage and retrieval"""
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory_file = f"user_memory_{user_id}.json"
        self.preferences = self.load_preferences()
        
    def load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading preferences: {e}")
                return self.get_default_preferences()
        return self.get_default_preferences()
    
    def get_default_preferences(self) -> Dict[str, Any]:
        """Return default user preferences structure"""
        return {
            "dietary_restrictions": [],
            "favorite_cuisines": [],
            "budget_range": {"min": 0, "max": 50},
            "preferred_meal_times": [],
            "disliked_foods": [],
            "order_history": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def save_preferences(self):
        """Save current preferences to file"""
        try:
            self.preferences["last_updated"] = datetime.now().isoformat()
            with open(self.memory_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def update_dietary_restrictions(self, restrictions: List[str]):
        """Update dietary restrictions"""
        self.preferences["dietary_restrictions"] = list(set(restrictions))
        self.save_preferences()
    
    def add_favorite_cuisine(self, cuisine: str):
        """Add a favorite cuisine"""
        if cuisine not in self.preferences["favorite_cuisines"]:
            self.preferences["favorite_cuisines"].append(cuisine)
            self.save_preferences()
    
    def set_budget_range(self, min_budget: float, max_budget: float):
        """Set budget range"""
        self.preferences["budget_range"] = {"min": min_budget, "max": max_budget}
        self.save_preferences()
    
    def add_to_order_history(self, order: Dict[str, Any]):
        """Add an order to history"""
        order["timestamp"] = datetime.now().isoformat()
        self.preferences["order_history"].append(order)
        # Keep only last 20 orders
        self.preferences["order_history"] = self.preferences["order_history"][-20:]
        self.save_preferences()
    
    def get_preference_summary(self) -> str:
        """Get a text summary of user preferences for the AI"""
        summary = f"""
User Preferences Summary:
- Dietary Restrictions: {', '.join(self.preferences['dietary_restrictions']) or 'None'}
- Favorite Cuisines: {', '.join(self.preferences['favorite_cuisines']) or 'Not specified'}
- Budget Range: ${self.preferences['budget_range']['min']} - ${self.preferences['budget_range']['max']}
- Disliked Foods: {', '.join(self.preferences['disliked_foods']) or 'None specified'}
- Recent Orders: {len(self.preferences['order_history'])} previous orders
"""
        return summary.strip()
    
    def extract_preferences_from_conversation(self, user_message: str):
        """Extract preferences from user conversation"""
        message_lower = user_message.lower()
        
        # Extract dietary restrictions
        dietary_keywords = {
            'vegetarian': 'vegetarian',
            'vegan': 'vegan',
            'gluten-free': 'gluten-free',
            'dairy-free': 'dairy-free',
            'halal': 'halal',
            'kosher': 'kosher',
            'keto': 'keto',
            'paleo': 'paleo'
        }
        
        for keyword, restriction in dietary_keywords.items():
            if keyword in message_lower:
                if restriction not in self.preferences["dietary_restrictions"]:
                    self.preferences["dietary_restrictions"].append(restriction)
        
        # Extract budget mentions
        budget_phrases = ['budget', 'spend', 'under', 'around', '$']
        if any(phrase in message_lower for phrase in budget_phrases):
            # Simple budget extraction (you can make this more sophisticated)
            import re
            budget_match = re.search(r'\$?(\d+)', user_message)
            if budget_match:
                budget = int(budget_match.group(1))
                if budget > self.preferences["budget_range"]["max"]:
                    self.preferences["budget_range"]["max"] = budget
        
        # Extract cuisine preferences
        cuisines = ['italian', 'chinese', 'indian', 'mexican', 'thai', 'japanese', 
                   'mediterranean', 'american', 'french', 'korean', 'vietnamese']
        
        for cuisine in cuisines:
            if cuisine in message_lower:
                if cuisine not in self.preferences["favorite_cuisines"]:
                    self.preferences["favorite_cuisines"].append(cuisine)
        
        self.save_preferences()