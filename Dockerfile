FROM python:3.12-slim

ENV HF_HOME=/models \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY server.py http_api.py .

VOLUME ["/models"]
EXPOSE 10300 8000
ENTRYPOINT ["python", "/app/server.py"]
