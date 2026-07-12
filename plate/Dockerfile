# Thai licence-plate — GPU (MIGraphX) build for the AMD server.
#
# The YOLO11s plate detector runs on MIGraphX: ~2.3 ms vs ~138 ms/CPU for a
# 640x640 pass (60x). It runs the exported ONNX directly (see plate_detector.py)
# because ultralytics' AutoBackend won't select the MIGraphX provider.
# The Thai OCR (PaddleOCR, PIR format) stays on CPU — small plate crops, and
# paddle2onnx does not cleanly convert the PIR checkpoint.
#
# ROCm userspace is bind-mounted from the host (/opt/rocm); onnxruntime-migraphx
# provides the onnxruntime namespace (installed last so it shadows any plain
# onnxruntime that ultralytics pulls in).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics YOLO_AUTOINSTALL=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libgl1 libglib2.0-0 libnuma1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY requirements.rocm.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
      https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

ENV LD_LIBRARY_PATH=/opt/rocm/lib
WORKDIR /app
COPY plate_tracker.py plate_preprocess.py upstream_pipeline.py ch.py service.py plate_detector.py /app/
COPY models /app/models

EXPOSE 8087
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8087"]
