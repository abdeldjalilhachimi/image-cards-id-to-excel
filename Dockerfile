FROM python:3.11-slim

WORKDIR /app

# System deps required by OpenCV headless and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models so startup is instant (avoids health-check timeout)
RUN python -c "import easyocr; easyocr.Reader(['ar', 'en'], gpu=False)"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
