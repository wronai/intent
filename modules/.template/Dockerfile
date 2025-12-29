# IntentForge Module Template
# Auto-generated module service

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy module code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV MODULE_PORT=8080

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]
