FROM python:3.12-slim
# ffmpeg: extract a single frame from a recording segment at a given time, so
# the playback review can look at surrounding cameras at the same moment.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ch.py .
COPY bridge.py .
EXPOSE 8081
CMD ["python", "bridge.py"]
