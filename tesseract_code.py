import cv2
import pytesseract
import re
import json
import os

def parse_menu_text(raw_text):
    items = []
    pattern = re.compile(
        r'^(?P<name>[\w\s&\',\-\.()+]+?)\s+'
        r'(?P<price>\d{2,4}\s?/?-?)$'
    )
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            name = match.group('name').strip()
            price = match.group('price').strip()
            items.append({'name': name, 'price': price})
    return items

# Image path
img_path = r'F:\AGENTIC\Team A\test_image\full_menu.png'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot find or open image at {img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(img_rgb, lang='eng', config='--oem 1 --psm 6')

print("RAW OCR TEXT:")
print(text)

# Parse menu items
menu_items = parse_menu_text(text)

print("\nPARSED MENU ITEMS:")
for item in menu_items:
    print(f"Item: {item['name']}, Price: {item['price']}")

# Directory to save JSON (create if not exists)
output_dir = r'F:\AGENTIC\Team A\extracted_menu'
os.makedirs(output_dir, exist_ok=True)

# JSON file path
json_file_path = os.path.join(output_dir, 'menu_data.json')

# Save parsed menu to JSON file
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(menu_items, f, indent=4, ensure_ascii=False)

print(f"\nMenu data saved to {json_file_path}")
