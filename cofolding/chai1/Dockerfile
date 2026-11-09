# syntax=docker/dockerfile:1.6
# Chai-1 0.6.1 on the same CUDA 13 / Python 3.12 base as the ESMFold2 image.
# chai_lab's metadata pins torch<2.7, which has no sm_120 kernels; the Blackwell-
# capable torch 2.11.0+cu130 wheel is installed first and chai_lab is installed
# with --no-deps.  See SPEC.md for this recorded divergence.
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
COPY requirements.lock /opt/requirements.lock
RUN python -m pip install -r /opt/requirements.lock
# TorchScript weights are loaded from the read-only asset mount; no download at run time.
RUN python -m pip install --no-deps chai_lab==0.6.1
# chai_lab is installed without its declared dependencies (its metadata pins
# torch<2.7, which cannot drive sm_120), so an incomplete lock would only surface
# as a runtime failure mid-fold. Import every submodule at build time instead.
COPY build_probe.py /tmp/build_probe.py
RUN python /tmp/build_probe.py && rm -f /tmp/build_probe.py
# Resident worker, throughput sweep, cloud transport double and the offline asset fetcher.
COPY resident /opt/cofold/resident
COPY fetch_assets.py assets.json throughput_benchmark.py cloud_job.py prepare_cloud_job.py /opt/cofold/
COPY cofold-throughput cofold-cloud-run /opt/cofold/
# Dependency-light shared helpers (numpy only) used for derived metrics.
# Build with --build-context common=cofolding/common.
COPY --from=common . /opt/cofold/cofolding/common
RUN chmod 755 /opt/cofold/resident/cofold-run \
 && ln -s /opt/cofold/resident/cofold-run /opt/venv/bin/cofold-run \
 && chmod 755 /opt/cofold/cofold-throughput /opt/cofold/cofold-cloud-run \
 && ln -s /opt/cofold/cofold-throughput /opt/venv/bin/cofold-throughput \
 && ln -s /opt/cofold/cofold-cloud-run /opt/venv/bin/cofold-cloud-run
ENV PYTHONPATH=/opt/cofold \
    CHAI_DOWNLOADS_DIR=/models
# The Ubuntu 24.04 base already defines uid/gid 1000 as `ubuntu`; creating another
# user with that id fails. Reuse it so mounted host files keep working at uid 1000.
RUN mkdir -p /workspace/cache /workspace/outputs /workspace/inputs /work \
 && chown -R ubuntu:ubuntu /workspace /work
ENV XDG_CACHE_HOME=/workspace/cache TORCH_HOME=/workspace/cache/torch \
    HF_HOME=/workspace/cache/hf HF_HUB_DISABLE_XET=1
# Image identity: the OCI label is the source of truth; the JSON mirror lets the
# worker inside the container read it without a Docker socket.
ARG CONTAINER_IMAGE_ID=unknown
RUN printf '{"image_id": "%s"}\n' "$CONTAINER_IMAGE_ID" > /opt/cofold/image-label.json
USER ubuntu
WORKDIR /work
ENTRYPOINT ["cofold-run"]
CMD ["--help"]
ARG CONTAINER_REVISION=unknown
ARG CONTAINER_VERSION=0.6.1-worker-rc1
LABEL org.opencontainers.image.title="Chai-1" \
      org.opencontainers.image.version="${CONTAINER_VERSION}" \
      org.opencontainers.image.revision="${CONTAINER_REVISION}" \
      org.opencontainers.image.source="https://github.com/AetherArk/container-library" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Chai-1 (chai_lab 0.6.1); Python 3.12; PyTorch 2.11.0/cu130; external model assets" \
      org.opencontainers.image.vendor="AetherArk" \
      ai.aetherark.chai1.source="https://github.com/chaidiscovery/chai-lab" \
      ai.aetherark.chai1.revision="unknown" \
      ai.aetherark.chai1.model_revision="chaiassets-chai1-inference-depencencies" \
      ai.aetherark.cofold.image_id="${CONTAINER_IMAGE_ID}"
