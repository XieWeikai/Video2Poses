FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:${PATH}
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app:/app/third_party/map-anything

WORKDIR /app

COPY pyproject.toml README.md main.py /app/
COPY video2poses /app/video2poses
COPY service /app/service
COPY configs /app/configs
COPY docker /app/docker
COPY third_party/map-anything /app/third_party/map-anything

RUN conda create -n service python=3.10 -y && \
    conda run -n service pip install --upgrade pip && \
    conda run -n service pip install \
      torch==2.6.0 \
      torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124 && \
    conda run -n service pip install \
      fastapi \
      pydantic \
      pyyaml \
      uvicorn && \
    conda run -n service pip install -e /app/third_party/map-anything && \
    conda run -n service pip install -e /app && \
    conda clean -afy

EXPOSE 18080

ENTRYPOINT ["/bin/bash", "/app/docker/entrypoint.sh"]
