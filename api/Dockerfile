# Используем базовый образ
FROM python:3.10

# Установка необходимых инструментов
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN apt-get update && apt-get install -y fonts-freefont-ttf fonts-dejavu && fc-cache -f -v


# Создаем рабочую директорию
WORKDIR /

# Копируем файл с необходимыми библиотеками для сервиса
COPY requirements.txt /

# Устанавливаем необходимые библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы необходимые для работы сервиса в контейнер
COPY . ./api

# Делаем порт 8087 видимым снаружи
EXPOSE 8098

# Запускаем сервис внутри контейнера
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8098"]
