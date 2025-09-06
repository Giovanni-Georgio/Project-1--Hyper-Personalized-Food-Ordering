from selenium import webdriver
from selenium.webdriver.chrome.service import Service

chromedriver_path = r"F:\AGENTIC\Team A\chromedriver-win64\chromedriver.exe"  # Use your exact path here
service = Service(executable_path=chromedriver_path)

driver = webdriver.Chrome(service=service)
driver.get("https://www.google.com")
print(driver.title)
driver.quit()
