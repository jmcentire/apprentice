FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Python dependencies — pinned versions for reproducibility
RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@2025.3.19" \
    "xformers==0.0.29.post3" \
    "trl>=0.7,<0.13" \
    "peft>=0.13,<0.15" \
    "accelerate>=0.34,<1.0" \
    "bitsandbytes>=0.44,<0.45" \
    "google-cloud-storage>=2.10,<3.0" \
    "sentencepiece>=0.2,<0.3" \
    "protobuf>=4.25,<6.0" \
    "datasets>=3.0,<4.0"

# Build llama.cpp for GGUF conversion — pinned to a release tag
ARG LLAMA_CPP_VERSION=b4722
RUN git clone --branch ${LLAMA_CPP_VERSION} --depth 1 \
        https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DGGML_CUDA=ON \
    && cmake --build build --config Release -j$(nproc) \
    && cp build/bin/llama-quantize /usr/local/bin/llama-quantize

ENV LLAMA_CPP_PATH=/usr/local/bin/llama-quantize

WORKDIR /app
COPY train.py /app/train.py

ENTRYPOINT ["python", "train.py"]
