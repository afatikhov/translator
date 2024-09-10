# Image Text Translator


#### Подготовил Фатихов Артур

Этот проект представляет собой сервис на базе FastAPI, который обрабатывает изображения, распознает текст с помощью OCR, переводит его и помещает переведенный текст обратно на изображение. Для распознавания текста используется EasyOCR, а для перевода — Google Translate. Сервис поддерживает обработку текста на английском и русском языках.

## Оглавление
1. Интерфейс 
2. Функции
3. Использование
   - Запуск сервиса
   - Тестирование
4. Краткое описание файлов в сервисе
5. Вывод


## Интерфейс

Сервис предназначен для запуска в виде Docker-контейнера и требует минимального количества файлов. При запуске будут установлены все необходимые библиотеки и загружены требуемые модели.
Приложение предоставляет API для взаимодействия с пользователем через один эндпоинт:

#### /translate_photo
Принимает 3 параметра:

- file: изображение в формате UploadFile, которое нужно обработать (например, JPG, PNG).
- lang_from: язык, с которого нужно перевести текст (например, 'en' для английского).
- lang_to: язык, на который нужно перевести текст (например, 'ru' для русского).

Для использования сервиса необходимо отправить POST-запрос к эндпоинту `/translate_photo`:

    # Загрузка изображения
    image = cv2.imread(source_path, cv2.IMREAD_COLOR)
    # print(image)
    # Создание данных для отправки на сервис
    data = {
        'file': cv2.imencode('.png', image)[1].tobytes(),
        'lang_from': 'en',
        'lang_to': 'ru'
    }
    # Отправка Пост запроса на сервис
    try:
        response = requests.post(url=url, files=data)
    except Exception as e:
        print(e)



## Функции

### textsize(text, font): 
    Определяет размеры текста для заданного шрифта. Используется для корректного размещения текста на изображении.

### find_contrasting_color(background_color): 
    Находит цвет текста, который будет контрастировать с цветом фона, чтобы обеспечить читаемость текста.

### draw_rectangle(image, points): 
    Переводит координаты прямоугольника из формата (x, y, width, height) в формат (x1, y1, x2, y2)

### find_dominant_color(image, top_left, bottom_right): 
    Определяет доминантный цвет внутри заданного прямоугольника на изображении.

### fill_rectanglar(image, x1, y1, x2, y2, color_uint8): 
    Закрашивает область прямоугольника на изображении заданным цветом, чтобы скрыть оригинальный текст.

### get_max_font_size(text, max_width, max_height, font_path):
    Рассчитывает максимальный размер шрифта, чтобы текст поместился в заданный прямоугольник.

### write_inside_a_box(image_array, x, y, w, h, text, color):
    Размещает текст внутри прямоугольника на изображении, используя указанный цвет и рассчитанный размер шрифта.

### process_image(file, lang_from, lang_to): 
    Основная функция FastAPI, которая обрабатывает загруженное изображение, распознает текст с помощью EasyOCR, переводит его с одного языка на другой, и отображает переведенный текст на изображении с учетом контрастных цветов. Возвращает переведенное изображение в формате PNG.


## Использование
### Запуск сервиса
1. Для запуска приложения нужно:

- перейти в папку translator

- создать и запустить docker container с помощью команды

sudo docker compose up

При запуске будет происходить сборка контейнера и установка необходимых библиотек.

### Тестирование
Для тестирования сервиса используйте файлы, находящиеся в папке translator/test:

1. Файл test.py предназначен для тестирования отправкой одного файла для перевода.
В разделе if __name__ == "__main__": можно задать путь к изображению и параментры:

'lang_from': - язык для распознавания и перевода на изображении
'lang_to': язык на который нужно перевести текст на изображении
Изображение с переведенным текстом сохранится в папку translator/test/translated c сохранением названия и добавлением пометки _translated. При необходимости путь можно изменить. Для запуска теста выполните команду:

python3 test.py

При успешном выполнении теста вы увидите сообщение о сохранении измененного изображения.

2. Файл test_all.py, при запуске которого все файлы из папки translator/test/original будут проведены через сервис, все новые изображения будут сохранены в translator/test/translated c сохранением названия и добавлением пометки _translated. Для запуска теста выполните команду:

python3 test_all.py

## Краткое описание файлов в сервисе 

'''
translator/
├── api
│   ├── arial.ttf               # Шрифт Arial для отображения текста на изображениях (поддерживает русский и английский языки)
│   ├── Dockerfile              # Файл для создания Docker-образа сервиса с настройкой окружения и зависимостей
│   ├── main.py                 # Основной код приложения на FastAPI для обработки изображений и перевода текста
│   └── requirements.txt        # Список зависимостей Python, необходимых для работы приложения
├── docker-compose.yml          # Файл для запуска сервиса и его зависимостей с помощью Docker Compose
├── readme.md                   # Документация проекта с инструкциями по установке, запуску и использованию сервиса
└── test
    ├── original                # Папка для хранения исходных изображений для тестирования
    ├── test_all.py             # Скрипт для запуска теста для всех файлов в original
    ├── test.py                 # Тесты для проверки функциональности приложения
    └── translated              # Папка для хранения переведенных изображений после тестирования
'''

# Вывод 
Этот проект предназначен для автоматического распознавания и перевода текста на изображениях. Он реализует использование EasyOCR для распознавания текста и Google Translate для перевода. Обработка изображений осуществляется с помощью OpenCV и Pillow, что обеспечивает гибкость работы с различными форматами и типами изображений.

Проект развертывается с помощью Docker, что упрощает установку, настройку окружения и обеспечивает изоляцию зависимостей. В будущем можно заменить используемые библиотеки на более производительные или собственные решения для улучшения качества распознавания текста, добавления поддержки других языков или настройки функциональности под специфические нужды.

Такой подход обеспечивает гибкость, масштабируемость и готовность проекта к адаптации под новые требования или технологии в области компьютерного зрения и обработки текста на изображениях.

