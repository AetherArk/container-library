# syntax=docker/dockerfile:1.6
# Protenix 2.0.0 on CUDA 12.8 / Python 3.12.  The pinned stack mirrors the working
# `local/protenix:2.0.0` image (torch 2.7.1+cu128, protenix 2.0.0, cuequivariance 0.8.0), which
# is the only configuration measured on the RTX 5070 Ti (sm_120) cards.  See SPEC.md §2.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
# `devel`, not `runtime`: protenix compiles its bundled fast-layer-norm CUDA kernel through
# torch.utils.cpp_extension, and gencode for sm_120 needs nvcc >= 12.8.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential ninja-build cmake git \
      ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN python -m pip install pip==26.0.1 setuptools==78.1.0 wheel==0.48.0
RUN python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu128
COPY requirements.lock /opt/requirements.lock
RUN python -m pip install -r /opt/requirements.lock
# Weights and the CCD/cluster cache are loaded from the read-only asset mount; nothing is
# downloaded at run time.  --no-deps: the lock above is the exact `pip freeze` of the proven
# environment, and protenix's own metadata would re-resolve it against current PyPI.
RUN python -m pip install --no-deps protenix==2.0.0
# Resident worker, throughput sweep, cloud transport double and the offline asset fetcher.
COPY resident /opt/cofold/resident
COPY fetch_assets.py assets.json throughput_benchmark.py cloud_job.py prepare_cloud_job.py build_probe.py /opt/cofold/
COPY cofold-throughput cofold-cloud-run /opt/cofold/
# Dependency-light shared helpers (numpy only) used for derived metrics.
# Build with --build-context common=cofolding/common.
COPY --from=common . /opt/cofold/cofolding/common
ENV PYTHONPATH=/opt/cofold
# Import every module the worker reaches and precompile the fast-layer-norm extension so no
# CUDA toolchain work happens inside a run.  An incomplete lock fails the build, not a fold.
# The bundled fast-layer-norm kernel is compiled here through torch.utils.cpp_extension.
# Without an explicit architecture list nvcc emits sm_70..sm_100 and the RTX 50-series cards
# (sm_120) then fail at run time with "no kernel image is available for execution".
ENV TORCH_CUDA_ARCH_LIST="12.0"
COPY build_kernel.py /opt/cofold/build_kernel.py
RUN python /opt/cofold/build_kernel.py && rm -f /opt/cofold/build_kernel.py
RUN python /opt/cofold/build_probe.py && rm -f /opt/cofold/build_probe.py
RUN chmod 755 /opt/cofold/resident/cofold-run \
 && ln -s /opt/cofold/resident/cofold-run /opt/venv/bin/cofold-run \
 && chmod 755 /opt/cofold/cofold-throughput /opt/cofold/cofold-cloud-run \
 && ln -s /opt/cofold/cofold-throughput /opt/venv/bin/cofold-throughput \
 && ln -s /opt/cofold/cofold-cloud-run /opt/venv/bin/cofold-cloud-run
# PROTENIX_ROOT_DIR is where protenix.data resolves checkpoint/protenix-v2.pt and common/.
# The mount is read-only at run time; every writable path lives under /workspace.
# CUTLASS_PATH is deliberately unset: the released protenix-v2 configuration uses the
# cuequivariance triangle kernels, and the deepspeed attention backend (which needs CUTLASS)
# is not part of this container's qualified scope.
ENV PROTENIX_ROOT_DIR=/models \
    TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions
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
ARG CONTAINER_VERSION=2.0.0-worker-rc1
LABEL org.opencontainers.image.title="Protenix" \
      org.opencontainers.image.version="${CONTAINER_VERSION}" \
      org.opencontainers.image.revision="${CONTAINER_REVISION}" \
      org.opencontainers.image.source="https://github.com/AetherArk/container-library" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Protenix 2.0.0 (protenix-v2); Python 3.12; PyTorch 2.7.1/cu128; external model assets" \
      org.opencontainers.image.vendor="AetherArk" \
      ai.aetherark.protenix.source="https://github.com/bytedance/Protenix" \
      ai.aetherark.protenix.revision="2.0.0" \
      ai.aetherark.protenix.model_revision="protenix-v2@653edab28103133512575365130916e3fd23ecc3" \
      ai.aetherark.cofold.image_id="${CONTAINER_IMAGE_ID}"
