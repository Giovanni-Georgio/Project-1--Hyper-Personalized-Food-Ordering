import cv2
import pytesseract

img_path = r'F:\AGENTIC\Team A\test_image\full_menu.png'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot find or open image at {img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

text = pytesseract.image_to_string(img_rgb, lang='eng', config='--oem 1 --psm 6')
print(text)
