FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL="C.UTF-8"
ENV TZ="UTC"
RUN apt-get update \
    && apt-get install --no-install-recommends --quiet --yes \
        curl \
        git \
        htop \
        less \
        libxrender1 \
        nvtop \
        openbabel \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

ENV UV_INSTALL_DIR="/usr/local/bin"
ENV UV_LINK_MODE="copy"
ENV VIRTUAL_ENV_DISABLE_PROMPT="1"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
