FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ReadMe.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY config.ini ./
COPY profiles ./profiles

ENTRYPOINT ["subtitlegen"]
CMD ["generate", "/data/videos", "--cache-dir", "/cache"]