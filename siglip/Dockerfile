# AMD/ROCm build of the siglip (ReID) service.
#
# The query API (/graph, /providers, the reid lists) is served from the
# transferred reid.db sqlite and needs no GPU — so the camera-graph page works
# the moment this starts. The identity embedder (youtureid.onnx, Apache-2.0)
# runs on the GPU through onnxruntime-migraphx. torch is CPU-only here: it is
# imported at module load, but the heavy SigLIP2 path (_load_siglip) is only
# taken when REID_EMBEDDER=siglip, and the default is youtureid.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libdrm-amdgpu1 libdrm2 libelf1 libnuma1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf libdrm_amdgpu.so.1 /usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so

# CPU torch (small) — only for imports and the non-default SigLIP2 path.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir \
        "transformers>=4.49.0" "fastapi>=0.110" "uvicorn[standard]>=0.29" \
        "pillow>=10.0" "requests>=2.31" "numpy>=1.26" "websocket-client>=1.8" \
        https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
WORKDIR /app
COPY . /app
EXPOSE 8085
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8085"]
