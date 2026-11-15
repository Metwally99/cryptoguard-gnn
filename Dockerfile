FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY pyproject.toml .
RUN python -m src.models.pipeline

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5).close()"]
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
