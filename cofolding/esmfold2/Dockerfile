# syntax=docker/dockerfile:1.6
# Reproduces the qualified ESMFold2 stack from local/esmfold2:3.4.1.post1-cu130:
# CUDA 13.0.2 devel base, Python 3.12 venv, torch 2.11.0+cu130,
# esm[fused] 3.4.1.post1, transformer-engine[pytorch] 2.19.0, gemmi.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential ninja-build cmake git \
      ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN python -m pip install pip==26.2.1 setuptools==78.1.0 wheel==0.48.0
RUN python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
# transformer_engine compiles its torch bindings against torch's bundled cuDNN/NCCL
# headers; these are the env conventions of the qualified build, not runtime knobs.
ENV CPATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/include:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/include \
    NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=120 MAX_JOBS=4
RUN python -m pip install --no-build-isolation 'transformer-engine[pytorch]==2.19.0'
COPY requirements.lock /opt/requirements.lock
RUN python -m pip install -r /opt/requirements.lock \
 && python -m pip check
# Resident worker, throughput sweep and the offline asset fetcher.
COPY resident /opt/cofold/resident
COPY batch_native /opt/cofold/batch_native
COPY fetch_assets.py assets.json throughput_benchmark.py cloud_job.py /opt/cofold/
COPY cofold-throughput cofold-cloud-run /opt/cofold/
# Dependency-light shared helpers (numpy only) used for derived metrics.
# Build with --build-context common=cofolding/common.
COPY --from=common . /opt/cofold/cofolding/common
RUN chmod 755 /opt/cofold/resident/cofold-run \
 && ln -s /opt/cofold/resident/cofold-run /opt/venv/bin/cofold-run \
 && chmod 755 /opt/cofold/cofold-throughput /opt/cofold/cofold-cloud-run \
 && ln -s /opt/cofold/cofold-throughput /opt/venv/bin/cofold-throughput \
 && ln -s /opt/cofold/cofold-cloud-run /opt/venv/bin/cofold-cloud-run
ENV PYTHONPATH=/opt/cofold
# The Ubuntu 24.04 base already defines uid/gid 1000 as `ubuntu`; creating another
# user with that id fails. Reuse it so mounted host files keep working at uid 1000.
RUN mkdir -p /workspace/cache /workspace/outputs /workspace/inputs /work \
 && chown -R ubuntu:ubuntu /workspace /work
ENV ESMFOLD2_CACHE=/workspace/cache TRITON_CACHE_DIR=/workspace/cache/triton \
    HF_HOME=/workspace/cache/hf XDG_CACHE_HOME=/workspace/cache TORCH_HOME=/workspace/cache/torch \
    HF_HUB_DISABLE_XET=1
# Image identity: the OCI label is the source of truth; the JSON mirror lets the
# worker inside the container read it without a Docker socket.
ARG CONTAINER_IMAGE_ID=unknown
RUN printf '{"image_id": "%s"}\n' "$CONTAINER_IMAGE_ID" > /opt/cofold/image-label.json
USER ubuntu
WORKDIR /work
ENTRYPOINT ["cofold-run"]
CMD ["--help"]
ARG CONTAINER_REVISION=unknown
ARG CONTAINER_VERSION=3.4.1.post1-worker-rc1
LABEL org.opencontainers.image.title="ESMFold2" \
      org.opencontainers.image.version="${CONTAINER_VERSION}" \
      org.opencontainers.image.revision="${CONTAINER_REVISION}" \
      org.opencontainers.image.source="https://github.com/AetherArk/container-library" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="ESMFold2 (esm 3.4.1.post1); Python 3.12; PyTorch 2.11.0/cu130; external model assets" \
      org.opencontainers.image.vendor="AetherArk" \
      ai.aetherark.esmfold2.source="https://github.com/Biohub/esm" \
      ai.aetherark.esmfold2.revision="43b4548b86762edfa747b07d5f440aad3c33acee" \
      ai.aetherark.esmfold2.model_revision="69869f737beffec5294845ede23db5fc0b4f509e" \
      ai.aetherark.cofold.image_id="${CONTAINER_IMAGE_ID}"
