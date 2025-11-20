FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source; yolov5 may or may not exist locally.
COPY . .

# If the yolov5 directory is missing or incomplete (not pushed to git),
# fetch the official repo during the image build so imports still work.
RUN if [ ! -d "yolov5/models" ]; then \
        echo "Local yolov5 directory missing; cloning from Ultralytics repo..." && \
        rm -rf yolov5 && \
        git clone --depth=1 https://github.com/ultralytics/yolov5.git yolov5; \
    fi

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--threads", "2", "--timeout", "600", "--worker-tmp-dir", "/dev/shm", "-k", "uvicorn.workers.UvicornWorker"]

