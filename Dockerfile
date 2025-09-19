# Use a non-slim, full Debian-based Python image to ensure all system libraries are present
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコンテナにコピー
COPY requirements.txt .

# requirements.txtを使ってpipでライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt
