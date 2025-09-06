from selenium.webdriver.chrome.service import Service  # add import

def test_selenium():
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),  # fixed here
            options=options
        )
        driver.get('https://www.google.com')
        driver.quit()
        print("✓ Selenium WebDriver working")
        return True
    except Exception as e:
        print(f"✗ Selenium test failed: {e}")
        print("  Make sure Google Chrome is installed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)  # removed the trailing text
