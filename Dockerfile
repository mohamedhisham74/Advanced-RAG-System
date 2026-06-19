FROM python:3.11-slim

# PyMuPDF needs libmupdf system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/     ./app/
COPY static/  ./static/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
