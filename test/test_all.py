from test import create_request
import os
from tqdm import tqdm

# Папка в которой находятся тестовые изображения
source_folder = 'original/'

# Папка, куда нужно сохранить все обработанные изображения
save_folder = 'translated/'

# Получаем все тестовые изображения
all_filenames = os.listdir(source_folder)

# Проходимся по всем тестовым изображениям
for filename in tqdm(all_filenames):
    # Получаем путь к исходному файлу
    source_path = os.path.join(source_folder, filename)
    filelist = source_path.split('/')[-1].split('.')
    filename = filelist[-2]
    extension = filelist[-1]
    # Получаем путь к файлу, который нужно сохранить
    save_filename = f'{filename}_translated.{extension}'
    save_path = os.path.join(save_folder, save_filename)
    # Направляем на функцию, которую используем для тестирования
    create_request(source_path, save_path)