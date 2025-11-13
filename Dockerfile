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
    git \
    libxrender1 \
    #以下2行geminiインストール用#
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    ###
    && rm -rf /var/lib/apt/lists/*

# Gemini CLI をグローバルインストール
RUN npm install -g @google/gemini-cli

ENV GEMINI_API_KEY = "AIzaSyDtPTdv09dxyLs59ZCftJu0TLpXbbN-M1k"

# requirements.txtをコンテナにコピー
COPY requirements.txt .

# requirements.txtを使ってpipでライブラリをインストール
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt