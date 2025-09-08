#!/usr/bin/env python3
"""
Startup script for the Hyper-Personalized Food Ordering AI
This script handles environment setup and launches the Streamlit app
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'langchain', 
        'huggingface_hub',
        'transformers',
        'torch',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing packages"""
    logger = logging.getLogger(__name__)
    
    for package in packages:
        logger.info(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            logger.info(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    logger = logging.getLogger(__name__)
    
    # Set Hugging Face token
    hf_token = "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"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    logger.info("✅ Hugging Face token configured")
    
    # Set other environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
    os.environ["ENVIRONMENT"] = "development"
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    logger.info("✅ Environment setup completed")

def check_file_structure():
    """Check if all required files are present"""
    logger = logging.getLogger(__name__)
    
    required_files = [
        "app.py",
        "agent_core.py", 
        "memory_store.py",
        "config.py"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    logger.info("✅ All required files present")
    return True

def test_imports():
    """Test if all modules can be imported successfully"""
    logger = logging.getLogger(__name__)
    
    try:
        from agent_core import FoodOrderingAgent
        from memory_store import UserMemory
        import config
        logger.info("✅ All modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def run_streamlit():
    """Launch the Streamlit application"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Streamlit application...")
        
        # Run streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false',
            '--server.headless', 'false'
        ])
        
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit: {e}")
        return False
    
    return True

def run_health_check():
    """Run a quick health check of the system"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Running system health check...")
    
    try:
        # Test agent initialization
        from agent_core import FoodOrderingAgent
        agent = FoodOrderingAgent()
        
        # Test memory system
        from memory_store import UserMemory
        memory = UserMemory("health_check_user")
        memory.update_dietary_restrictions(["vegetarian"])
        
        # Clean up test data
        import os
        test_file = "user_memory_health_check_user.json"
        if os.path.exists(test_file):
            os.remove(test_file)
        
        logger.info("✅ Health check passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False

def main():
    """Main function to start the application"""
    logger = setup_logging()
    
    print("=" * 60)
    print("🍕 Hyper-Personalized Food Ordering AI")
    print("=" * 60)
    print("Setting up your AI food assistant...")
    print()
    
    # Step 1: Check file structure
    logger.info("Step 1: Checking file structure...")
    if not check_file_structure():
        print("❌ Missing required files. Please ensure all files are in the correct location.")
        sys.exit(1)
    print("✅ File structure OK")
    
    # Step 2: Check dependencies
    logger.info("Step 2: Checking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("Installing missing dependencies...")
        if not install_dependencies(missing_deps):
            print("❌ Failed to install dependencies")
            sys.exit(1)
    print("✅ Dependencies OK")
    
    # Step 3: Setup environment
    logger.info("Step 3: Setting up environment...")
    setup_environment()
    print("✅ Environment OK")
    
    # Step 4: Test imports
    logger.info("Step 4: Testing imports...")
    if not test_imports():
        print("❌ Import test failed")
        sys.exit(1)
    print("✅ Imports OK")
    
    # Step 5: Health check
    logger.info("Step 5: Running health check...")
    if not run_health_check():
        print("⚠️  Health check failed, but continuing anyway...")
    else:
        print("✅ Health check OK")
    
    print()
    print("🎉 Setup completed successfully!")
    print("🚀 Starting the application...")
    print()
    print("📝 Note: The app will open in your default browser")
    print("💡 If it doesn't open automatically, go to: http://localhost:8501")
    print("⌨️  Press Ctrl+C to stop the application")
    print()
    
    # Step 6: Launch application
    if not run_streamlit():
        print("❌ Failed to start application")
        sys.exit(1)

if __name__ == "__main__":
    main()