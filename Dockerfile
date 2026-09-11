# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG PRELOAD_MODELS=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/huggingface \
    TORCH_HOME=/opt/models/torch \
    NLTK_DATA=/opt/models/nltk \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt \
    && python -m pip check

COPY download_models.py ./
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python download_models.py; fi

COPY speech_pipeline.py batch_pipeline.py yt_transcriber_api.py smoke_test.py container_entrypoint.sh ./

RUN python -m py_compile speech_pipeline.py batch_pipeline.py yt_transcriber_api.py download_models.py smoke_test.py \
    && python smoke_test.py

RUN mkdir -p /data/input /data/results /data/work \
    && chmod -R a+rX /opt/models \
    && chmod -R a+rwX /data \
    && chmod +x /app/container_entrypoint.sh

VOLUME ["/data/results", "/data/work"]
EXPOSE 8000

ENTRYPOINT ["/app/container_entrypoint.sh"]
CMD ["api"]
