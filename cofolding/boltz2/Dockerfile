# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:59e0e4376a0f16d10b03d3a14344b80a866a1674cb4948cb318291387ac05010

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates git \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN python -m pip install pip==26.0.1 setuptools==82.0.1 wheel==0.46.3
COPY requirements.lock /opt/requirements.lock
RUN python -m pip install -r /opt/requirements.lock
# Fetch the exact source; no pre-existing checkout or unrecorded local patches.
RUN git init /opt/boltz \
 && git -C /opt/boltz remote add origin https://github.com/jwohlwend/boltz.git \
 && git -C /opt/boltz fetch --depth 1 origin cb04aeccdd480fd4db707f0bbafde538397fa2ac \
 && git -C /opt/boltz checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/boltz rev-parse HEAD)" = cb04aeccdd480fd4db707f0bbafde538397fa2ac \
 && python -m pip install --no-deps --no-build-isolation '/opt/boltz[cuda]' \
 && python -m pip check \
 && rm -rf /opt/boltz/.git
# Keep the native boltz CLI; add the common persistent-worker entrypoint.
COPY resident /opt/cofold/resident
COPY batch_native /opt/cofold/batch_native
COPY cloud_job.py fetch_assets.py assets.json /opt/cofold/
COPY cofold-cloud-run /opt/cofold/cofold-cloud-run
RUN chmod 755 /opt/cofold/cofold-cloud-run \
 && ln -s /opt/cofold/cofold-cloud-run /opt/venv/bin/cofold-cloud-run
RUN chmod 755 /opt/cofold/resident/cofold-run \
 && ln -s /opt/cofold/resident/cofold-run /opt/venv/bin/cofold-run
ENV PYTHONPATH=/opt/cofold
RUN groupadd --gid 1000 boltz \
 && useradd --uid 1000 --gid 1000 --create-home boltz \
 && mkdir -p /workspace/cache /workspace/outputs /workspace/inputs /work \
 && chown -R boltz:boltz /workspace /work
ENV BOLTZ_CACHE=/workspace/cache TRITON_CACHE_DIR=/workspace/cache/triton \
    HF_HOME=/workspace/cache/hf XDG_CACHE_HOME=/workspace/cache
USER boltz
WORKDIR /work
ENTRYPOINT ["cofold-run"]
CMD ["--help"]
ARG CONTAINER_REVISION=unknown
ARG CONTAINER_VERSION=2.2.1-worker-v3-benchmark-rc1
LABEL org.opencontainers.image.title="Boltz2" \
      org.opencontainers.image.version="${CONTAINER_VERSION}" \
      org.opencontainers.image.revision="${CONTAINER_REVISION}" \
      org.opencontainers.image.source="https://github.com/AetherArk/container-library" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="Boltz 2.2.1; Python 3.11; PyTorch 2.7.1/cu128; external model assets" \
      org.opencontainers.image.vendor="AetherArk" \
      ai.aetherark.boltz.source="https://github.com/jwohlwend/boltz" \
      ai.aetherark.boltz.revision="cb04aeccdd480fd4db707f0bbafde538397fa2ac"
