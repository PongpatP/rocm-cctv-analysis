# Open-vocabulary grounding (text -> boxes) on CPU. Torch is the CPU wheel on
# purpose: one frame per question, and the GPU stays free for detection + VLM.
FROM python:3.12-slim

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir transformers pillow fastapi uvicorn

WORKDIR /app
COPY ground.py /app/ground.py
ENTRYPOINT ["uvicorn", "ground:app", "--host", "0.0.0.0", "--port", "8090"]
