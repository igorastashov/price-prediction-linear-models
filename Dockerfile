FROM python:3.11-slim

# Устанавливаем рабочий каталог
WORKDIR /web-service

# Копируем все файлы из текущего каталога внутрь контейнера
COPY . .

# Устанавливаем пакетом unzip
RUN apt-get update && apt-get install -y unzip

# Устанавливаем права выполнения для скриптов
RUN chmod +x /web-service/data/download_data.sh
RUN chmod +x /web-service/fastapi/models/download_model.sh
RUN chmod +x /web-service/fastapi/weights/download_transformers.sh

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Выполняем скрипты при запуске контейнера
CMD bash -c "cd /web-service/data && ./download_data.sh && cd /web-service/fastapi/models && ./download_model.sh && cd /web-service/fastapi/weights && ./download_transformers.sh && cd /web-service/fastapi && \
    gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind=0.0.0.0:8000"