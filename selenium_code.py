import base64
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# Setup Chrome driver path
service = Service(executable_path='F:/AGENTIC/Team A/chromedriver-win64/chromedriver.exe')
driver = webdriver.Chrome(service=service)

# Open the target menu page
driver.get('https://www.shudhrestaurant.com/menu.html')

def fullpage_screenshot(driver, file):
    # Get page dimensions
    metrics = driver.execute_cdp_cmd('Page.getLayoutMetrics', {})
    width = metrics['contentSize']['width']
    height = metrics['contentSize']['height']

    # Override device metrics to capture full page
    driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
        "mobile": False,
        "width": width,
        "height": height,
        "deviceScaleFactor": 1,
        "screenOrientation": {"angle": 0, "type": "portraitPrimary"}
    })

    # Capture screenshot (Base64-encoded)
    screenshot = driver.execute_cdp_cmd('Page.captureScreenshot', {'fromSurface': True})
    # Decode and save to file
    with open(file, 'wb') as f:
        f.write(base64.b64decode(screenshot['data']))

    # Clear device metrics override
    driver.execute_cdp_cmd('Emulation.clearDeviceMetricsOverride', {})

# Save full page screenshot to this location
fullpage_screenshot(driver, 'F:/AGENTIC/Team A/test_image/full_menu.png')

driver.quit()
