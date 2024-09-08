from fastapi import FastAPI, UploadFile, HTTPException, Response
import cv2
import numpy as np
import pytesseract
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
import io
import uvicorn
import os
from collections import Counter
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'


app = FastAPI()

# Убедитесь, что путь к Tesseract указан правильно
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Путь к Tesseract

translator = Translator()


def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def find_contrasting_color(background_color):
    """
    Находит цвет текста, который будет контрастировать с фоном.

    :param background_color: Цвет фона в формате (R, G, B)
    :return: Цвет текста в формате (R, G, B)
    """
    # Преобразование цвета фона в массив NumPy
    bg_color = np.array(background_color)

    # Преобразование фона в цветовое пространство HSV
    bg_hsv = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_RGB2HSV)[0][0]

    # Прямой контрастный цвет в HSV пространстве
    text_hue = (bg_hsv[0] + 180) % 180
    text_saturation = 255
    text_value = 255 - bg_hsv[2]

    # Преобразование контрастного цвета обратно в RGB
    text_hsv = np.array([text_hue, text_saturation, text_value])
    text_rgb = cv2.cvtColor(np.uint8([[text_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

    # Конвертация в целочисленный формат
    return tuple(map(int, text_rgb))


def draw_rectangle(image, points):
    x1, y1, w, h = points
    x2, y2 = x1 + w, y1 + h
    # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return x1, y1, x2, y2


def find_dominant_color(image, top_left, bottom_right):
    # Извлечение области интереса (ROI)
    x1, y1 = top_left
    x2, y2 = bottom_right
    roi = image[y1:y2, x1:x2]

    # Преобразование ROI в двумерный массив пикселей
    pixels = roi.reshape(-1, 3)

    # Подсчет частоты каждого цвета
    pixel_counts = Counter(map(tuple, pixels))

    # Нахождение самого частого цвета
    most_common_color, _ = pixel_counts.most_common(1)[0]
    print(f'most common color: {most_common_color}')
    return most_common_color

def fill_rectanglar(image, x1, y1, x2, y2, color_uint8):
    color_int = tuple(int(value) for value in color_uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color_int, thickness=cv2.FILLED)


def write_inside_a_box_old(image_array, x, y, w, h, text, color=(255,255,255)):
    # Copy the image to avoid modifying the original
    output_image = image_array.copy()

    # Set up font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # Determine the maximum font scale that fits the text within the box
    while True:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        if text_width <= w and text_height <= h:
            break
        font_scale -= 0.1
        if font_scale <= 0:
            raise ValueError("Text is too large to fit in the given box")

    # Calculate text position to be centered in the box
    text_x = x + (w - text_width) // 2
    text_y = y + (h + text_height) // 2

    # Draw the text
    cv2.putText(output_image, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    return output_image


def get_max_font_size(text, max_width, max_height, font_path="arial.ttf"):
    """
    Determines the maximum font size for the given text to fit inside the box.

    Args:
        text (str): The text to be fitted.
        max_width (int): Maximum width of the box.
        max_height (int): Maximum height of the box.
        font_path (str): Path to the font file that supports Russian characters.

    Returns:
        int: Maximum font size that allows the text to fit inside the box.
    """
    # Initialize font size
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)

    while True:
        # Create an image to measure the text size
        image = Image.new('RGB', (max_width, max_height))
        draw = ImageDraw.Draw(image)
        text_width, text_height = textsize(text, font=font)

        # Check if the text fits inside the box
        if text_width > max_width or text_height > max_height:
            break

        font_size += 1
        font = ImageFont.truetype(font_path, font_size)

    return font_size - 1

def write_inside_a_box(image_array, x, y, w, h, text, color=(255,255,255)):
    # Convert OpenCV image to Pillow image
    pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Choose a font that supports Russian characters
    # You may need to specify the path to a font file that supports Russian
    font_path = "arial.ttf"  # You can use another font file if needed
    font_size = get_max_font_size(text, w, h)
    font = ImageFont.truetype(font_path, font_size)

    # Determine the maximum font size that fits the text within the box
    while True:
        text_width, text_height = textsize(text, font=font)
        if text_width <= w and text_height <= h:
            break
        font_size -= 1
        if font_size <= 0:
            raise ValueError("Text is too large to fit in the given box")
        font = ImageFont.truetype(font_path, font_size)

    # Calculate text position to be centered in the box
    text_x = x + (w - text_width) // 2
    text_y = y + (h - text_height) // 2

    # Draw the text
    draw.text((text_x, text_y), text, font=font, fill=color)

    # Convert Pillow image back to OpenCV format
    output_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return output_image



@app.post('/translate_photo')
async def process_image(file: UploadFile, lang_from: str = 'eng', lang_to: str = 'rus') -> Response:
    # Проверка, является ли загруженный файл изображением
    # if not file.content_type.startswith('image/'):
    #     raise HTTPException(status_code=400, detail="File is not an image")

    # Чтение изображения
    img_bytes = await file.read()  # Здесь нужно использовать await
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    # image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.COLOR_BGR2GRAY)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to read the image")

    # Преобразование изображения в формат PIL для дальнейшей работы с текстом
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Извлечение текста с изображения с использованием Tesseract OCR
    data = pytesseract.image_to_data(pil_image, lang='eng', output_type=pytesseract.Output.DICT)

    # Print out the extracted data
    # print(data)

    # Iterate over each word and its bounding box coordinates
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Filter out low-confidence results, adjust threshold as needed
            try:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                x1, y1, x2, y2 = draw_rectangle(image, (x, y, w, h))
                color = find_dominant_color(image, (x1, y1), (x2, y2))
                fill_rectanglar(image, x1, y1, x2, y2, color)
                text = data['text'][i]
                translated_text = translator.translate(text, src='en', dest='ru').text
                print(translated_text)
                text_color = find_contrasting_color(color)
                image = write_inside_a_box(image, x, y, w, h, translated_text, color=text_color)
                print(f"Text: {data['text'][i]} | Position: ({x}, {y}, {w}, {h}) | Confidence: {data['conf'][i]}")
            except Exception as e:
                print(e)
                # raise e
                break
    # if not extracted_text.strip():
    #     raise HTTPException(status_code=400, detail="No text found in the image")
    #
    # # Перевод текста с помощью Google Translate API
    # translated_text = translator.translate(extracted_text, src=lang_from, dest=lang_to).text
    #
    # # Наложение переведенного текста на изображение
    # draw = ImageDraw.Draw(pil_image)
    # font = ImageFont.truetype("arial.ttf", size=20)  # Убедитесь, что шрифт доступен на вашем сервере
    #
    # # Наложение текста на изображение (координаты и стиль можно настроить)
    # draw.text((10, 10), translated_text, fill="black", font=font)
    #
    # # Конвертация изображения обратно в формат OpenCV для отправки
    # processed_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    #
    # # Кодирование изображения в формат PNG
    processed_image = image
    _, encoded_image = cv2.imencode('.png', processed_image)
    # print(image)
    return Response(content=encoded_image.tobytes(), status_code=200, media_type='image/png')


if __name__ == '__main__':
    uvicorn.run('main_test:app', host='0.0.0.0', port=8098, workers=1, reload=False)