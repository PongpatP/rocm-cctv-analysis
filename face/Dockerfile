# Face recognition — GPU (MIGraphX) build for AMD server.
#
# The 112x112 cvlface embedder runs on MIGraphX: ~8ms for a batch of 8 vs
# ~79ms on CPU (10x). MediaPipe (alignment) stays CPU-only. ROCm userspace is
# bind-mounted from the host (/opt/rocm), same as airocm — the image carries
# no ROCm, keeping it small and ABI-matched to the host driver.
#
# onnxruntime-migraphx provides the onnxruntime namespace; do NOT also install
# plain onnxruntime (it would shadow the MIGraphX provider). cp312 = Ubuntu 24.04.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libgl1 libglib2.0-0 libgles2 libegl1 libopengl0 libxrender1 \
        libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY requirements.rocm.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
      https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

ENV LD_LIBRARY_PATH=/opt/rocm/lib
WORKDIR /app
COPY facerec.py /app/facerec.py
COPY ch.py /app/ch.py
COPY service.py /app/service.py
COPY models /app/models

EXPOSE 8086
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8086"]
