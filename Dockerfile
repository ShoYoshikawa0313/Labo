# Use a non-slim, full Debian-based Python image to ensure all system libraries are present
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    git \
    screen \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコンテナにコピー
COPY requirements.txt .

# uvをインストールし、uvを使ってrequirements.txtからライブラリをインストール
RUN pip install --no-cache-dir uv && \
    uv pip install --no-cache --system -r requirements.txt
