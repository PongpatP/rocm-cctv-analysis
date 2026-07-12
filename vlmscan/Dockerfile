# VLM preprocessing of recordings — captions person-active minutes so the
# investigator can search what happened, not just that someone was there.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir requests

WORKDIR /app
COPY scan.py /app/scan.py
ENTRYPOINT ["python", "-u", "/app/scan.py"]
