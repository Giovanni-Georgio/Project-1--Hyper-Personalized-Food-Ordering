import streamlit as st
import sys
import os
import json
import requests
from datetime import datetime
from agent_core import FoodOrderingAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🍕 Hyper-Personalized Food Ordering AI",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Giovanni-Georgio/Project-1--Hyper-Personalized-Food-Ordering',
        'Report a bug': "https://github.com/Giovanni-Georgio/Project-1--Hyper-Personalized-Food-Ordering/issues",
        'About': "# Hyper-Personalized Food Ordering AI\nBuilt with ❤️ using Streamlit, LangChain, and Hugging Face"
    }
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left-color: #667eea;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-left-color: #f093fb;
        margin-right: 2rem;
    }
    
    .preference-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff8a65;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc8 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffa8a8 0%, #ffd3d3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .order-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #ff9800;
        margin: 0.8rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e3f2fd;
        padding: 0.6rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "agent" not in st.session_state:
        try:
            # Initialize with your HF token
            st.session_state.agent = FoodOrderingAgent(huggingface_api_token="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")
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            st.error("Failed to initialize AI agent. Please check your setup.")
            st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "order_history" not in st.session_state:
        st.session_state.order_history = []
    
    if "api_status" not in st.session_state:
        st.session_state.api_status = check_api_status()

def check_api_status():
    """Check status of teammate APIs"""
    status = {
        "menu_api": False,
        "backend_api": False
    }
    
    try:
        # Check Team Member A's menu API
        response = requests.get("http://localhost:8001/health", timeout=2)
        status["menu_api"] = response.status_code == 200
    except:
        pass
    
    try:
        # Check Team Member B's backend API
        response = requests.get("http://localhost:8002/health", timeout=2)
        status["backend_api"] = response.status_code == 200
    except:
        pass
    
    return status

def display_message(message, is_user=False, timestamp=None):
    """Display a chat message with enhanced styling"""
    time_str = timestamp or datetime.now().strftime("%H:%M")
    
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>👤 You</strong>
                <small>{time_str}</small>
            </div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>🤖 Food AI Assistant</strong>
                <small>{time_str}</small>
            </div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)

def display_api_status():
    """Display API status indicators"""
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.api_status["menu_api"]:
                st.markdown('<div class="success-box">✅ Menu API Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ Menu API Offline (Using Sample Data)</div>', unsafe_allow_html=True)
        
        with col2:
            if st.session_state.api_status["backend_api"]:
                st.markdown('<div class="success-box">✅ Backend API Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ Backend API Offline (Mock Orders)</div>', unsafe_allow_html=True)
        
        with col3:
            try:
                if hasattr(st.session_state.agent, 'llm') and st.session_state.agent.llm:
                    st.markdown('<div class="success-box">✅ AI Model Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box">⚠️ AI Model: Fallback Mode</div>', unsafe_allow_html=True)
            except:
                st.markdown('<div class="error-box">❌ AI Model Error</div>', unsafe_allow_html=True)

def display_order_history():
    """Display user's order history"""
    preferences = st.session_state.agent.get_user_preferences()
    order_history = preferences.get("order_history", [])
    
    if order_history:
        st.subheader("📋 Recent Orders")
        for i, order in enumerate(reversed(order_history[-5:])):  # Show last 5 orders
            st.markdown(f"""
            <div class="order-card">
                <strong>🍽️ {order.get('item', 'Unknown Item')}</strong><br>
                <small>📍 {order.get('restaurant', 'Unknown Restaurant')} | 💰 ${order.get('price', 0)} | 🆔 {order.get('order_id', 'N/A')}</small><br>
                <small>📅 {order.get('timestamp', 'Unknown time')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No previous orders yet. Place your first order to see history here!")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🍕 Hyper-Personalized Food Ordering AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your intelligent food companion powered by advanced AI</p>', unsafe_allow_html=True)
    
    # API Status
    display_api_status()
    
    # Sidebar for preferences and controls
    with st.sidebar:
        st.header("🎯 Your Food Profile")
        
        # Get current preferences
        current_prefs = st.session_state.agent.get_user_preferences()
        
        # User Profile Section
        with st.expander("👤 User Profile", expanded=True):
            user_name = st.text_input("Name", value="Food Lover", help="Your name for personalized experience")
            user_location = st.text_input("Location", value="City Center", help="Delivery location")
        
        # Dietary Preferences
        st.subheader("🥗 Dietary Preferences")
        dietary_options = [
            "vegetarian", "vegan", "gluten-free", "dairy-free", 
            "halal", "kosher", "keto", "paleo", "low-carb", "nut-free"
        ]
        selected_dietary = st.multiselect(
            "Select dietary restrictions:",
            dietary_options,
            default=current_prefs.get("dietary_restrictions", []),
            help="Choose all that apply to filter menu options"
        )
        
        # Budget Range
        st.subheader("💰 Budget Settings")
        budget_col1, budget_col2 = st.columns(2)
        with budget_col1:
            budget_min = st.number_input(
                "Min Budget ($)",
                min_value=5,
                max_value=200,
                value=current_prefs.get("budget_range", {}).get("min", 10),
                step=5
            )
        with budget_col2:
            budget_max = st.number_input(
                "Max Budget ($)",
                min_value=10,
                max_value=500,
                value=current_prefs.get("budget_range", {}).get("max", 30),
                step=5
            )
        
        # Cuisine Preferences
        st.subheader("🌍 Cuisine Preferences")
        cuisine_options = [
            "Italian", "Chinese", "Indian", "Mexican", "Thai", "Japanese",
            "Mediterranean", "American", "French", "Korean", "Vietnamese",
            "Greek", "Lebanese", "Spanish", "Healthy", "Fast Food"
        ]
        selected_cuisines = st.multiselect(
            "Favorite cuisines:",
            cuisine_options,
            default=current_prefs.get("favorite_cuisines", []),
            help="Select your preferred cuisines"
        )
        
        # Meal Timing
        st.subheader("⏰ Meal Preferences")
        meal_times = st.multiselect(
            "Preferred meal times:",
            ["Breakfast", "Brunch", "Lunch", "Dinner", "Late Night", "Snacks"],
            default=current_prefs.get("preferred_meal_times", [])
        )
        
        # Update preferences
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Profile", use_container_width=True):
                st.session_state.agent.user_memory.update_dietary_restrictions(selected_dietary)
                st.session_state.agent.user_memory.set_budget_range(budget_min, budget_max)
                for cuisine in selected_cuisines:
                    st.session_state.agent.user_memory.add_favorite_cuisine(cuisine.lower())
                st.session_state.agent.user_memory.preferences["preferred_meal_times"] = meal_times
                st.session_state.agent.user_memory.save_preferences()
                st.success("✅ Profile updated!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.agent.clear_conversation()
                st.success("✅ Chat cleared!")
                st.rerun()
        
        # Display current preferences
        st.subheader("📊 Current Profile")
        with st.expander("View Details", expanded=True):
            st.markdown(f"""
            <div class="preference-box">
                <strong>🥗 Dietary:</strong> {', '.join(current_prefs.get('dietary_restrictions', [])) or 'None'}<br>
                <strong>💰 Budget:</strong> ${current_prefs.get('budget_range', {}).get('min', 0)} - ${current_prefs.get('budget_range', {}).get('max', 50)}<br>
                <strong>🌍 Cuisines:</strong> {', '.join(current_prefs.get('favorite_cuisines', [])) or 'All'}<br>
                <strong>📋 Total Orders:</strong> {len(current_prefs.get('order_history', []))}<br>
                <strong>⏰ Meal Times:</strong> {', '.join(current_prefs.get('preferred_meal_times', [])) or 'Anytime'}
            </div>
            """, unsafe_allow_html=True)
        
        # Order History
        display_order_history()
        
        # Quick Stats
        st.subheader("📈 Your Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(current_prefs.get('order_history', []))}</h3>
                <p>Total Orders</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_spend = 0
            orders = current_prefs.get('order_history', [])
            if orders:
                avg_spend = sum(order.get('price', 0) for order in orders) / len(orders)
            st.markdown(f"""
            <div class="metric-card">
                <h3>${avg_spend:.1f}</h3>
                <p>Avg. Spend</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat with Your AI Food Assistant")
        
        # Chat container
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                welcome_message = f"""Welcome, {user_name}! 👋 

I'm your hyper-personalized AI food assistant. I've learned about your preferences and I'm ready to help you discover amazing meals that perfectly match your taste, budget, and dietary needs.

🎯 **What I know about you:**
- Budget: ${budget_min} - ${budget_max}
- Dietary: {', '.join(selected_dietary) if selected_dietary else 'No restrictions'}
- Favorite Cuisines: {', '.join(selected_cuisines) if selected_cuisines else 'Open to all'}

**What would you like to do today?**
- 🔍 Find restaurants and dishes
- 🛒 Place an order
- 💡 Get personalized recommendations
- 📊 Explore new cuisines

Just start chatting naturally - I understand context and learn from our conversation!"""
                
                display_message(welcome_message, is_user=False)
            
            # Display conversation history
            for message in st.session_state.messages:
                display_message(
                    message["content"], 
                    message["role"] == "user",
                    message.get("timestamp")
                )
        
        # Chat input
        with st.container():
            user_input = st.chat_input(
                "💬 Ask me anything about food... (e.g., 'I want healthy Italian under $20')",
                key="chat_input"
            )
        
        # Process user input
        if user_input:
            timestamp = datetime.now().strftime("%H:%M")
            
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input,
                "timestamp": timestamp
            })
            
            # Show thinking indicator
            with st.spinner("🤔 Analyzing your preferences and finding perfect options..."):
                try:
                    response = st.session_state.agent.process_message(user_input)
                    
                    # Add AI response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                except Exception as e:
                    error_response = f"""I apologize, but I encountered an error while processing your request: {str(e)}

🔧 **Troubleshooting suggestions:**
- Try rephrasing your question
- Check if the AI model is properly loaded
- Verify your internet connection

Please try asking again, and I'll do my best to help you find great food options!"""
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
            
            st.rerun()
    
    with col2:
        st.subheader("🚀 Quick Actions")
        
        # Smart Quick Actions based on preferences
        quick_actions = [
            ("🍕 Italian Favorites", "Show me the best Italian dishes within my budget"),
            ("🥗 Healthy Options", "I want healthy, nutritious meals"),
            ("💰 Budget Meals", "Show me the best value meals"),
            ("🌶️ Spicy Food", "I'm craving something spicy"),
            ("🥘 Surprise Me", "Recommend something new based on my taste profile"),
            ("⚡ Quick Delivery", "What can I get delivered fastest?")
        ]
        
        for button_text, query in quick_actions:
            if st.button(button_text, use_container_width=True):
                timestamp = datetime.now().strftime("%H:%M")
                
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": timestamp
                })
                
                with st.spinner("Processing..."):
                    response = st.session_state.agent.process_message(query)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                
                st.rerun()
        
        # Help Section
        st.subheader("💡 Smart Tips")
        with st.expander("How to get the best results", expanded=False):
            st.markdown("""
            **🎯 Be Specific:**
            - "Vegetarian Italian under $15"
            - "Quick healthy breakfast options"
            
            **🗣️ Chat Naturally:**
            - "I'm feeling like Thai food tonight"
            - "Something comforting for this rainy day"
            
            **📱 Order Commands:**
            - "Order the Margherita Pizza"
            - "Add the Buddha Bowl to my cart"
            
            **🔄 Get Updates:**
            - "Update my budget to $40"
            - "I'm now following a keto diet"
            """)
        
        # System Info
        st.subheader("ℹ️ System Info")
        with st.expander("Technical Details", expanded=False):
            st.markdown(f"""
            **🤖 AI Model:** {"Connected" if hasattr(st.session_state.agent, 'llm') and st.session_state.agent.llm else "Fallback Mode"}
            
            **📊 APIs Status:**
            - Menu API: {"🟢 Online" if st.session_state.api_status["menu_api"] else "🔴 Offline"}
            - Backend API: {"🟢 Online" if st.session_state.api_status["backend_api"] else "🔴 Offline"}
            
            **💾 Memory:** {len(st.session_state.messages)} messages stored
            
            **🔄 Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
            """)
            
            if st.button("🔄 Refresh System Status"):
                st.session_state.api_status = check_api_status()
                st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.markdown("""
        **🔧 Troubleshooting:**
        1. Make sure all dependencies are installed
        2. Check your Hugging Face token
        3. Verify your Python environment
        4. Try refreshing the page
        
        **📞 Need Help?**
        Check the [GitHub repository](https://github.com/Giovanni-Georgio/Project-1--Hyper-Personalized-Food-Ordering) for support.
        """)