from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
import os
import requests
import json
from typing import Dict, Any, List, Optional
from memory_store import UserMemory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOutputParser(BaseOutputParser):
    """Simple parser to clean up LLM outputs"""
    
    def parse(self, text: str) -> str:
        # Clean up the response
        lines = text.strip().split('\n')
        # Take the first meaningful line that's not empty
        for line in lines:
            if line.strip() and not line.startswith('Human:') and not line.startswith('Assistant:'):
                return line.strip()
        return text.strip()

class FoodOrderingAgent:
    """AI Agent for personalized food ordering assistance with team integration"""
    
    def __init__(self, huggingface_api_token: str = None):
        # Initialize user memory
        self.user_memory = UserMemory()
        
        # Set up Hugging Face token
        self.hf_token = huggingface_api_token or "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"
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_token
        
        # Initialize conversation memory
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="user_input"
        )
        
        # Initialize LLM with error handling
        self.llm = self._initialize_llm()
        
        # Create the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create the chain if LLM is available
        if self.llm:
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
                memory=self.conversation_memory,
                output_parser=SimpleOutputParser(),
                verbose=False
            )
        else:
            self.chain = None
            logger.warning("LLM not available, using fallback responses")
        
        # Mock API endpoints (will be replaced with real teammate APIs)
        self.menu_api_base = "http://localhost:8001"  # Team Member A's menu API
        self.backend_api_base = "http://localhost:8002"  # Team Member B's backend API
        
        # Sample restaurant data (fallback when APIs not available)
        self.sample_restaurants = self._load_sample_restaurants()
    
    def _initialize_llm(self) -> Optional[HuggingFaceHub]:
        """Initialize the Hugging Face LLM with error handling"""
        try:
            # Try multiple free models in order of preference
            models_to_try = [
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill",
                "google/flan-t5-base",
                "microsoft/DialoGPT-small"
            ]
            
            for model_id in models_to_try:
                try:
                    logger.info(f"Trying to initialize model: {model_id}")
                    llm = HuggingFaceHub(
                        repo_id=model_id,
                        model_kwargs={
                            "temperature": 0.7,
                            "max_length": 200,
                            "do_sample": True
                        },
                        huggingfacehub_api_token=self.hf_token
                    )
                    # Test the model with a simple query
                    test_response = llm("Hello")
                    logger.info(f"Successfully initialized model: {model_id}")
                    return llm
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_id}: {e}")
                    continue
            
            logger.error("All models failed to initialize")
            return None
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return None
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for the food agent"""
        template = """You are a helpful food ordering assistant. Help users find perfect meals based on their preferences.

User Preferences: {user_preferences}

Available Menu Items: {menu_data}

Previous Conversation: {chat_history}

User: {user_input}
Assistant: I'll help you find the perfect meal! """
        
        return PromptTemplate(
            input_variables=["user_preferences", "menu_data", "chat_history", "user_input"],
            template=template
        )
    
    def _load_sample_restaurants(self) -> List[Dict[str, Any]]:
        """Load sample restaurant data (fallback)"""
        return [
            {
                "id": "rest_001",
                "name": "Mario's Italian Bistro",
                "cuisine": "Italian",
                "rating": 4.5,
                "delivery_time": "25-35 min",
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
                        "ingredients": ["mozzarella", "basil", "tomato", "flour", "olive oil"]
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
                        "ingredients": ["chicken", "breadcrumbs", "marinara", "mozzarella"]
                    }
                ]
            },
            {
                "id": "rest_002",
                "name": "Spice Garden Indian",
                "cuisine": "Indian",
                "rating": 4.3,
                "delivery_time": "30-40 min",
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
                        "ingredients": ["basmati rice", "mixed vegetables", "spices", "saffron"]
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
                        "ingredients": ["chicken", "tomato", "cream", "butter", "spices"]
                    }
                ]
            },
            {
                "id": "rest_003",
                "name": "Green Bowl Healthy",
                "cuisine": "Healthy",
                "rating": 4.6,
                "delivery_time": "20-30 min",
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
                        "ingredients": ["quinoa", "sweet potato", "broccoli", "avocado", "tahini"]
                    }
                ]
            }
        ]
    
    def get_menu_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get menu data from Team Member A's API or fallback to sample data"""
        try:
            # Try to call Team Member A's menu API
            response = requests.get(
                f"{self.menu_api_base}/api/menu",
                params=filters or {},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as e:
            logger.warning(f"Menu API not available: {e}")
        
        # Fallback to sample data
        return self.sample_restaurants
    
    def filter_menu_by_preferences(self, menu_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter menu items based on user preferences"""
        preferences = self.user_memory.preferences
        budget_max = preferences["budget_range"]["max"]
        dietary_restrictions = preferences["dietary_restrictions"]
        favorite_cuisines = preferences["favorite_cuisines"]
        
        filtered_restaurants = []
        
        for restaurant in menu_data:
            # Check cuisine preference
            cuisine_match = (not favorite_cuisines or 
                           any(cuisine.lower() in restaurant.get("cuisine", "").lower() 
                               for cuisine in favorite_cuisines))
            
            if cuisine_match:
                filtered_items = []
                for item in restaurant.get("items", []):
                    # Check budget
                    if item.get("price", 0) <= budget_max:
                        # Check dietary restrictions
                        item_suitable = True
                        
                        for restriction in dietary_restrictions:
                            if restriction == "vegetarian" and not item.get("vegetarian", False):
                                item_suitable = False
                                break
                            elif restriction == "vegan" and not item.get("vegan", False):
                                item_suitable = False
                                break
                            elif restriction == "gluten-free" and not item.get("gluten_free", False):
                                item_suitable = False
                                break
                        
                        if item_suitable:
                            filtered_items.append(item)
                
                if filtered_items:
                    restaurant_copy = restaurant.copy()
                    restaurant_copy["items"] = filtered_items
                    filtered_restaurants.append(restaurant_copy)
        
        return filtered_restaurants
    
    def place_order_via_api(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place order via Team Member B's backend API"""
        try:
            response = requests.post(
                f"{self.backend_api_base}/api/orders",
                json=order_data,
                timeout=10
            )
            if response.status_code in [200, 201]:
                return response.json()
        except requests.RequestException as e:
            logger.warning(f"Backend API not available: {e}")
        
        # Mock response for testing
        return {
            "order_id": "ORD_" + str(hash(str(order_data)))[-8:],
            "status": "confirmed",
            "estimated_delivery": "30-40 minutes",
            "total": order_data.get("total", 0),
            "message": "Order placed successfully! (Mock response)"
        }
    
    def process_message(self, user_input: str) -> str:
        """Process user message and return agent response"""
        try:
            # Extract preferences from conversation
            self.user_memory.extract_preferences_from_conversation(user_input)
            
            # Get menu data
            menu_data = self.get_menu_data()
            
            # Filter by preferences
            filtered_menu = self.filter_menu_by_preferences(menu_data)
            
            # Get user preferences summary
            preferences_summary = self.user_memory.get_preference_summary()
            
            # Try to use LLM if available
            if self.chain:
                try:
                    response = self.chain.run({
                        "user_input": user_input,
                        "user_preferences": preferences_summary,
                        "menu_data": self._format_menu_for_llm(filtered_menu[:2])  # Limit for token efficiency
                    })
                    return response
                except Exception as e:
                    logger.warning(f"LLM processing failed: {e}")
            
            # Fallback to rule-based response
            return self._generate_rule_based_response(user_input, filtered_menu, preferences_summary)
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return "I apologize, but I encountered an error. Could you please try rephrasing your request?"
    
    def _format_menu_for_llm(self, restaurants: List[Dict[str, Any]]) -> str:
        """Format menu data for LLM consumption"""
        if not restaurants:
            return "No restaurants match your current preferences."
        
        formatted = []
        for restaurant in restaurants[:3]:  # Limit for token efficiency
            items_text = []
            for item in restaurant.get("items", [])[:3]:  # Limit items per restaurant
                items_text.append(f"- {item['name']}: ${item['price']} ({item['description']})")
            
            if items_text:
                formatted.append(f"{restaurant['name']} ({restaurant.get('cuisine', 'Various')}):\n" + "\n".join(items_text))
        
        return "\n\n".join(formatted)
    
    def _generate_rule_based_response(self, user_input: str, filtered_menu: List[Dict[str, Any]], preferences: str) -> str:
        """Generate rule-based response when LLM is not available"""
        user_input_lower = user_input.lower()
        
        # Handle ordering intent
        if any(word in user_input_lower for word in ['order', 'buy', 'purchase', 'get this', 'i want this']):
            return self._handle_order_intent(user_input, filtered_menu)
        
        # Handle greeting
        if any(greeting in user_input_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
            return f"""Hello! 👋 I'm your AI food assistant, ready to help you find the perfect meal!

{preferences}

I can help you:
🔍 Find restaurants and dishes that match your preferences
💰 Stay within your budget
🥗 Filter by dietary restrictions
📱 Place orders through our integrated system

What are you in the mood for today?"""

        # Handle budget queries
        elif any(word in user_input_lower for word in ['budget', 'cheap', 'affordable', 'price', 'cost']):
            budget_options = self._get_budget_friendly_options(filtered_menu)
            return f"""Here are some great budget-friendly options within your ${self.user_memory.preferences['budget_range']['max']} limit:

{budget_options}

Would you like to see more details about any of these dishes or place an order?"""

        # Handle cuisine-specific queries
        elif any(cuisine in user_input_lower for cuisine in ['italian', 'indian', 'chinese', 'healthy', 'mexican']):
            cuisine_options = self._get_cuisine_options(user_input_lower, filtered_menu)
            return f"""Great choice! Here are some delicious options matching your cuisine preference:

{cuisine_options}

Which one catches your eye? I can provide more details or help you place an order!"""

        # Default recommendation response
        else:
            recommendations = self._format_recommendations(filtered_menu)
            return f"""Based on your preferences, here are my top recommendations:

{recommendations}

💡 **Pro tip**: You can ask me to:
- "Show me vegetarian options under $15"
- "Order the Margherita Pizza"
- "What's the healthiest option?"
- "I want something spicy"

What sounds good to you?"""
    
    def _handle_order_intent(self, user_input: str, menu_data: List[Dict[str, Any]]) -> str:
        """Handle order placement intent"""
        # Simple item matching (in production, this would be more sophisticated)
        for restaurant in menu_data:
            for item in restaurant.get("items", []):
                if item["name"].lower() in user_input.lower():
                    order_data = {
                        "user_id": "default_user",
                        "restaurant_id": restaurant.get("id", "unknown"),
                        "items": [{"item_id": item.get("id", "unknown"), "quantity": 1}],
                        "total": item["price"],
                        "preferences": self.user_memory.preferences
                    }
                    
                    order_result = self.place_order_via_api(order_data)
                    
                    # Add to order history
                    self.user_memory.add_to_order_history({
                        "restaurant": restaurant["name"],
                        "item": item["name"],
                        "price": item["price"],
                        "order_id": order_result.get("order_id", "unknown")
                    })
                    
                    return f"""🎉 Order Confirmed!

📋 **Order Details:**
- {item['name']} from {restaurant['name']}
- Price: ${item['price']}
- Order ID: {order_result.get('order_id', 'N/A')}

⏰ **Estimated Delivery:** {order_result.get('estimated_delivery', '30-40 minutes')}

Your order has been placed successfully! You'll receive updates on the delivery status.

Need anything else? I can help you add more items or track your order."""

        return """I'd be happy to help you place an order! Could you please specify which dish you'd like? 

For example, you can say:
- "I want the Margherita Pizza"
- "Order me the Buddha Bowl"
- "Get me the Butter Chicken"

Or let me know if you'd like to see more menu options first!"""
    
    def _get_budget_friendly_options(self, menu_data: List[Dict[str, Any]]) -> str:
        """Get budget-friendly menu options"""
        budget_items = []
        budget_max = self.user_memory.preferences["budget_range"]["max"]
        
        for restaurant in menu_data[:3]:
            for item in restaurant.get("items", []):
                if item["price"] <= budget_max:
                    budget_items.append(f"💰 **{item['name']}** - ${item['price']}\n   📍 {restaurant['name']}\n   📝 {item['description']}")
        
        return "\n\n".join(budget_items[:5]) if budget_items else "No options found within your budget. Would you like to adjust your budget range?"
    
    def _get_cuisine_options(self, user_query: str, menu_data: List[Dict[str, Any]]) -> str:
        """Get cuisine-specific options"""
        cuisine_items = []
        
        for restaurant in menu_data:
            restaurant_cuisine = restaurant.get("cuisine", "").lower()
            if any(cuisine in user_query for cuisine in [restaurant_cuisine] if cuisine):
                for item in restaurant.get("items", [])[:3]:
                    cuisine_items.append(f"🍽️ **{item['name']}** - ${item['price']}\n   📍 {restaurant['name']} ({restaurant.get('cuisine', 'Various')})\n   📝 {item['description']}")
        
        return "\n\n".join(cuisine_items) if cuisine_items else "No cuisine-specific options found. Let me show you our full menu!"
    
    def _format_recommendations(self, menu_data: List[Dict[str, Any]]) -> str:
        """Format menu recommendations"""
        if not menu_data:
            return """No restaurants currently match your preferences. Here are some options:

🔧 **Adjust Your Preferences:**
- Increase your budget range
- Try different cuisines
- Modify dietary restrictions

Or ask me for "show me all options" to see everything available!"""
        
        recommendations = []
        for restaurant in menu_data[:3]:
            for item in restaurant.get("items", [])[:2]:
                recommendations.append(f"""🌟 **{item['name']}** - ${item['price']}
📍 {restaurant['name']} ({restaurant.get('cuisine', 'Various')})
⭐ {restaurant.get('rating', 'N/A')} stars | ⏱️ {restaurant.get('delivery_time', 'N/A')}
📝 {item['description']}""")
        
        return "\n\n".join(recommendations)
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get current user preferences"""
        return self.user_memory.preferences
    
    def update_preferences(self, **kwargs):
        """Update user preferences"""
        for key, value in kwargs.items():
            if key in self.user_memory.preferences:
                self.user_memory.preferences[key] = value
        self.user_memory.save_preferences()
    
    def clear_conversation(self):
        """Clear conversation memory"""
        if self.conversation_memory:
            self.conversation_memory.clear()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        if hasattr(self.conversation_memory, 'chat_memory'):
            messages = []
            for message in self.conversation_memory.chat_memory.messages:
                messages.append({
                    "type": message.__class__.__name__,
                    "content": message.content
                })
            return messages
        return []