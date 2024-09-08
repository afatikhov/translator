import numpy as np
import requests
import cv2

url = 'http://0.0.0.0:8098/translate_photo'

def create_request(source_path, save_path):
    image = cv2.imread(source_path, cv2.IMREAD_COLOR)
    print(image)
    data = {
        'file': cv2.imencode('.png', image)[1].tobytes(),
        # 'content_type': 'image/'
    }
    try:
        response = requests.post(url=url, files=data)
    except Exception as e:
        print(e)


    if response.status_code == 200:
        image_translated = np.asarray(bytearray(response.content), dtype='uint8')
        image_translated = cv2.imdecode(image_translated, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, image_translated)
    else:
        print(f"Request failed with status code {response.status_code}")


if __name__ == '__main__':
    # source_path = 'original/Slide31articlestext-over-images.png'
    source_path = 'original/IvV2y.png'
    filelist = source_path.split('/')[-1].split('.')
    filename = filelist[-2]
    extention = filelist[-1]
    save_path = f'translated/{filename}_translated.{extention}'
    create_request(source_path, save_path)