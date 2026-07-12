# The AMD `ai` container.
#
# ROCm is not installed into the image: /opt/rocm is bind-mounted from the host
# (same Ubuntu 24.04 ABI), which keeps this image ~400 MB instead of ~30 GB and
# guarantees the userspace matches the host's amdgpu driver exactly.
#
# onnxruntime-migraphx is AMD's only ONNX Runtime build for ROCm 7.2 — there is
# no ROCMExecutionProvider wheel. cp312 matches this base image's python.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --no-cache-dir pyyaml pillow \
      https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# The migraphx wheel provides the onnxruntime namespace. Ultralytics' auto-install
# would pull the CPU-only onnxruntime and BREAK MIGraphX. Disable auto-install:
ENV YOLO_AUTOINSTALL=false

ENV LD_LIBRARY_PATH=/opt/rocm/lib
WORKDIR /app
COPY airocm.py /app/airocm.py
ENTRYPOINT ["python3", "/app/airocm.py"]
