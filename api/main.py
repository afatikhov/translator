from fastapi import FastAPI, UploadFile, HTTPException, Response
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from collections import Counter
import easyocr
from googletrans import Translator

app = FastAPI()

# Создаем объект переводчика
translator = Translator()


def textsize(text, font):
    '''
    Эта функция необходима для определения размеров текста,
    который будет нарисован с использованием данного шрифта.
    '''
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def find_contrasting_color(background_color):
    """
    Эта функция находит цвет текста, который будет контрастировать с фоном.
    Принимает аргументы:
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
    '''
    Эта функция
    :param image: изображение на котором необходимо нарисовать квадрат
    :param points: точки в виде x1, y1, w, h
    :return: точки в формате x1, y1, x2, y2
    '''
    x1, y1, w, h = points
    # Перевод точек из одного формата в другой
    x2, y2 = x1 + w, y1 + h
    # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return x1, y1, x2, y2


def find_dominant_color(image, top_left, bottom_right):
    '''
    Эта функция находит доминантный цвет внутри квадрата, где находится надпись.
    Принимает аргументы:
    :param image: изображение
    :param top_left: крайняя левая точка (x, y)
    :param bottom_right: крайняя правая точка (x, y)
    :return: самый частый цвет внутри квадрата с надписью
    '''
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
    '''
    Эта функция закрашивает область, где находился текст для перевода.
    :param image: изображение на котором нужно закрасить область np.array
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param color_uint8: цвет в формате uint8
    :return: None
    '''
    color_int = tuple(int(value) for value in color_uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color_int, thickness=cv2.FILLED)



def get_max_font_size(text, max_width, max_height, font_path="api/arial.ttf"):
    """
    Эта функция определяет максимальный размер шрифта для текста, который может поместиться внутри заданного прямоугольника.
    :text (str): Текст, который нужно разместить.
    :max_width (int): Максимальная ширина прямоугольника.
    :max_height (int): Максимальная высота прямоугольника.
    :font_path (str): Путь к файлу шрифта, поддерживающего русские символы.
    :returns (int): Максимальный размер шрифта, который позволяет тексту поместиться в прямоугольнике.
    """
    # Изначальный размер шрифта
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)

    while True:
        # Создание изображения для измерения размера текста
        image = Image.new('RGB', (max_width, max_height))
        draw = ImageDraw.Draw(image)
        text_width, text_height = textsize(text, font=font)

        # Проверка, помещается ли текст внутри прямоугольника
        if text_width > max_width or text_height > max_height:
            break

        # Если вмещается увеличиваем шрифт на один
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)

    # Когда шрифт перестал помещаться возвращаем шрифт со значением -1
    return font_size - 1

def write_inside_a_box(image_array, x, y, w, h, text, color=(255,255,255)):
    '''
    Эта функция пишет внутри прямоугольника откуда был взять и переведен текст.
    Принимает аргументы
    :param image_array: изображение
    :param x: координата левого верхнего угла
    :param y: координата левого верхнего угла
    :param w: ширина квадрата
    :param h: высота квадрата
    :param text: текст для написания
    :param color: цвет
    :return:
    '''
    # Конвертация изображения OpenCV в изображение Pillow
    pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Выбор шрифта, поддерживающего русские символы
    font_path = "api/arial.ttf"
    font_size = get_max_font_size(text, w, h)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = textsize(text, font=font)

    # Вычисление позиции текста для центрирования в прямоугольнике
    text_x = x + (w - text_width) // 2
    text_y = y + (h - text_height) // 2

    # Рисование текста
    draw.text((text_x, text_y), text, font=font, fill=color)

    # Конвертация изображения Pillow обратно в формат OpenCV
    output_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return output_image


# Инициализация EasyOCR для распознавания текста на английском и русском языках
reader = easyocr.Reader(['en', 'ru'])


@app.post('/translate_photo')
async def process_image(file: UploadFile, lang_from: str = 'en', lang_to: str = 'ru') -> Response:
    '''
    Эта функция предназначения для использования с fastapi, принимает запрос на сервис, переводит
    надписи на изображении с использованием дополнительных функций. Возвращает переведенное изображение.
    Принимает аргументы:
    :param file: файл для обработки
    :param lang_from: язык в формате 'en', 'ru' с которого нужно сделать перевод
    :param lang_to: язык в формате 'en', 'ru' на который нужно сделать перевод
    :return: возвращает переведенное изображение
    '''
    # Чтение изображения
    img_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to read the image")

    # Конвертация изображения в формат PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Извлечение текста с помощью EasyOCR
    result = reader.readtext(np.array(pil_image), detail=True)

    # Обработка каждого обнаруженного текста и его координат
    for detection in result:
        bbox = detection[0]
        text = detection[1]
        confidence = detection[2]

        if confidence > 0.6:  # Порог уверенности, при котором текст будет обработан
            try:
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                w, h = x2 - x1, y2 - y1

                # Отрисовка прямоугольника вокруг обнаруженного текста
                x1, y1, x2, y2 = draw_rectangle(image, (x1, y1, w, h))

                # Определение доминирующего цвета и контрастного цвета
                color = find_dominant_color(image, (x1, y1), (x2, y2))
                fill_rectanglar(image, x1, y1, x2, y2, color)

                # Перевод текста
                translated_text = translator.translate(text, src=lang_from, dest=lang_to).text

                # Поиск контрастного цвета для текста
                text_color = find_contrasting_color(color)

                # Написание переведенного текста внутри прямоугольника
                image = write_inside_a_box(image, x1, y1, w, h, translated_text, color=text_color)

                print(f"Текст: {text} | Позиция: ({x1}, {y1}, {x2}, {y2}) | Уровень уверенности: {confidence}")
            except Exception as e:
                print(e)
                break

    # Конвертация изображения обратно в формат OpenCV
    processed_image = image
    _, encoded_image = cv2.imencode('.png', processed_image)

    # Возврат ответа на fastapi
    return Response(content=encoded_image.tobytes(), status_code=200, media_type='image/png')

if __name__ == '__main__':
    # Запуск прослушивания порта на uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8098, workers=1, reload=False)