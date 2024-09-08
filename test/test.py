import numpy as np
import requests
import cv2

# URL на который нужно отправить запрос для тестирования
url = 'http://0.0.0.0:8098/translate_photo'

def create_request(source_path, save_path):
    '''Эта функция загружает изображение с помощью cv2, из данного пути,
    создает запрос на сервис, затем после получения ответа создает новый файл с
    переведенным текстом, принимает аргументы:
    source_path: str, путь к изображению, которое необходимо перевести,
    save_path: str, путь, куда сохранить изображение'''
    # Загрузка изображения
    image = cv2.imread(source_path, cv2.IMREAD_COLOR)
    # print(image)
    # Создание данных для отправки на сервис
    data = {
        'file': cv2.imencode('.png', image)[1].tobytes(),
        # 'content_type': 'image/'
    }
    # Отправка Пост запроса на сервис
    try:
        response = requests.post(url=url, files=data)
    except Exception as e:
        print(e)

    # Проверка кода ответа
    if response.status_code == 200:
        # Перевод изображения из ответа в формат cv2 и его сохранение в save_path
        image_translated = np.asarray(bytearray(response.content), dtype='uint8')
        image_translated = cv2.imdecode(image_translated, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, image_translated)
    else:
        print(f"Request failed with status code {response.status_code}")


if __name__ == '__main__':
    # source_path = 'original/Slide31articlestext-over-images.png'
    # Используем путь начального изображения для создания пути для сохранения
    source_path = 'original/Slide31articlestext-over-images.png'
    filelist = source_path.split('/')[-1].split('.')
    filename = filelist[-2]
    extention = filelist[-1]
    save_path = f'translated/{filename}_translated.{extention}'
    # Отправка на функцию тестирования
    create_request(source_path, save_path)