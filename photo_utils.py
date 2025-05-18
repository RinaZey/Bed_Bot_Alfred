from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pytesseract
import cv2
import numpy as np

def blur_text(image_path, out_path):
    image = cv2.imread(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='rus')
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            roi = image[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(roi, (23, 23), 30)
            image[y:y+h, x:x+w] = blur
    cv2.imwrite(out_path, image)
    return out_path

def add_text(image_path, text, out_path, pos=(20,20)):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 32)
    draw.text(pos, text, (255,0,0), font=font)
    image.save(out_path)
    return out_path

def add_image(image_path, overlay_path, out_path, pos=(20,20)):
    image = Image.open(image_path)
    overlay = Image.open(overlay_path)
    image.paste(overlay, pos, overlay)
    image.save(out_path)
    return out_path

def apply_filter(image_path, out_path, filter_name="BLUR"):
    image = Image.open(image_path)
    if filter_name == "BLUR":
        image = image.filter(ImageFilter.BLUR)
    elif filter_name == "CONTOUR":
        image = image.filter(ImageFilter.CONTOUR)
    elif filter_name == "DETAIL":
        image = image.filter(ImageFilter.DETAIL)
    elif filter_name == "EMBOSS":
        image = image.filter(ImageFilter.EMBOSS)
    image.save(out_path)
    return out_path
