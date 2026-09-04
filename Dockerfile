FROM python:3.14.7-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    CREDITRISK_DATA_DIR=/app/data

WORKDIR /app

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/
COPY params/ ./params/
COPY tools/ ./tools/
COPY train.py tune.py predict.py explain.py evaluate.py update_readme.py ./

RUN useradd --create-home --uid 1000 creditrisk \
 && mkdir -p /app/data /app/models /app/predictions /app/reports /app/mlruns \
 && chown -R creditrisk:creditrisk /app
USER creditrisk

VOLUME ["/app/data", "/app/models", "/app/predictions", "/app/reports", "/app/mlruns"]

ENTRYPOINT ["python"]
CMD ["train.py"]
