# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG APT_PACKAGES="ffmpeg"
ARG PIP_EXTRAS=""

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    SUBTITLEGEN_CACHE_DIR=/cache \
    PADDLE_PDX_CACHE_HOME=/models/paddle \
    FLAGS_use_mkldnn=0 \
    FLAGS_enable_mkldnn=0 \
    FLAGS_enable_pir_api=0 \
    FLAGS_enable_pir_in_executor=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ${APT_PACKAGES} \
    && rm -rf /var/lib/apt/lists/*

# Third-party wheels depend only on pyproject extras. Source edits reuse this layer.
COPY pyproject.toml ReadMe.md scripts/docker_install_deps.py ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install setuptools wheel \
    && python docker_install_deps.py ${PIP_EXTRAS}

COPY src ./src
COPY profiles ./profiles
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --no-build-isolation .

COPY config.ini ./
ENTRYPOINT ["subtitlegen"]
