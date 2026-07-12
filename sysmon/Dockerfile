FROM python:3.12-slim
WORKDIR /app
# rocm-smi (bind-mounted from the host) needs libdrm to talk to /dev/kfd.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libdrm-amdgpu1 libdrm2 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf libdrm_amdgpu.so.1 /usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so
RUN pip install --no-cache-dir fastapi uvicorn psutil
COPY sysmon.py .
EXPOSE 8083
CMD ["python", "sysmon.py"]
