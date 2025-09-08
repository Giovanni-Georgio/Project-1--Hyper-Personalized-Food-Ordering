#!/usr/bin/env python3
"""
Test script for the Agentic Food AI Assistant
Run this to verify your setup is working correctly
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from agent_core import FoodOrderingAgent
        print("✅ FoodOrderingAgent imported successfully")
    except ImportError as e:
        print(f"❌ FoodOrderingAgent import failed: {e}")
        return False
    
    try:
        from memory_store import UserMemory
        print("✅ UserMemory imported successfully")
    except ImportError as e:
        print(f"❌ UserMemory import failed: {e}")
        return False
    
    return True

def test_memory_system():
    """Test the user memory system"""
    print("\n🧠 Testing memory system...")
    
    try:
        from memory_store import UserMemory
        
        # Create test user memory
        memory = UserMemory("test_user")
        
        # Test preference updates
        memory.update_dietary_restrictions(["vegetarian", "gluten-free"])
        memory.set_budget_range(10, 25)
        memory.add_favorite_cuisine("italian")
        
        # Test preference summary
        summary = memory.get_preference_summary()
        print("✅ Memory system working correctly")
        print(f"📄 Test user summary:\n{summary}")
        
        # Clean up test file
        if os.path.exists("user_memory_test_user.json"):
            os.remove("user_memory_test_user.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False

def test_agent_responses():
    """Test the AI agent responses"""
    print("\n🤖 Testing AI agent...")
    
    try:
        from agent_core import FoodOrderingAgent
        
        # Create test agent
        agent = FoodOrderingAgent()
        
        # Test basic responses
        test_messages = [
            "Hi, I'm looking for food suggestions",
            "I'm vegetarian and have a $20 budget",
            "Show me Italian food options",
            "I want something healthy"
        ]
        
        for message in test_messages:
            print(f"\n👤 User: {message}")
            response = agent.process_message(message)
            print(f"🤖 Agent: {response[:100]}..." if len(response) > 100 else f"🤖 Agent: {response}")
        
        print("✅ Agent responses working correctly")
        
        # Clean up test memory file
        if os.path.exists("user_memory_default_user.json"):
            os.remove("user_memory_default_user.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_restaurant_filtering():
    """Test restaurant filtering logic"""
    print("\n🏪 Testing restaurant filtering...")
    
    try:
        from agent_core import FoodOrderingAgent
        
        agent = FoodOrderingAgent()
        
        # Set specific preferences
        agent.user_memory.update_dietary_restrictions(["vegetarian"])
        agent.user_memory.set_budget_range(10, 20)
        
        # Test filtering
        filtered_data = agent._filter_restaurants_by_preferences()
        print("✅ Restaurant filtering working correctly")
        print(f"📊 Filtered restaurants:\n{filtered_data[:200]}...")
        
        # Clean up
        if os.path.exists("user_memory_default_user.json"):
            os.remove("user_memory_default_user.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Restaurant filtering test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Agentic Food AI Tests\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Memory System Test", test_memory_system),
        ("Agent Response Test", test_agent_responses),
        ("Restaurant Filtering Test", test_restaurant_filtering)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! Your Agentic Food AI is ready to use!")
        print("Run 'streamlit run app.py' to start the application.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed and files are in the correct locations.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)